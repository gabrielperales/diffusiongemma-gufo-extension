#!/usr/bin/env python3
"""Pack NVIDIA's DiffusionGemma NVFP4 checkpoint into one GGUF for Gufo.

`gufo serve llm` opens text models as GGUF files. This writes the decoder
tensors of nvidia/diffusiongemma-26B-A4B-it-NVFP4 byte for byte, under their
checkpoint names, into a GGUF whose metadata (architecture, tokenizer, chat
template, sampler settings) is copied from a llama.cpp "diffusion-gemma"
GGUF of the same model, such as AlexAtomic/diffusiongemma-26B-A4B-it-GGUF.
Only the metadata is read from that file.

Tensor types: BF16 and F32 keep their GGUF types; the packed E2M1 values
(U8) and their E4M3 block scales (F8_E4M3) are stored as GGUF I8 bytes. A
loader tells them apart by name: `*.weight_scale` is E4M3, every other I8
tensor is packed E2M1. The script refuses a checkpoint where that rule does
not hold. Vision tensors (model.encoder.vision_tower, embed_vision) belong in
the mmproj file and the FP4 activation scales (*.input_scale) are unused, so
neither is written. rope_freqs.weight, which llama.cpp GGUFs carry for the
global layers' proportional RoPE, is copied from the metadata GGUF.

Every tensor is hashed while it is copied and checked again after writing.

    python convert_nvfp4_gguf.py --checkpoint DIR --metadata-from REF.gguf \
        --out OUT.gguf [--source-revision SHA]
"""

import argparse
import hashlib
import json
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
# GGUF metadata value types.
T_UINT32, T_STRING, T_ARRAY, T_UINT64 = 4, 8, 9, 10
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}
# ggml tensor types.
GGML_F32, GGML_BF16, GGML_I8 = 0, 30, 24
FROM_SAFETENSORS = {"F32": (GGML_F32, 4), "BF16": (GGML_BF16, 2),
                    "U8": (GGML_I8, 1), "F8_E4M3": (GGML_I8, 1)}
WEIGHTS_FORMAT = "nvidia-modelopt-nvfp4-v1"
CHUNK = 64 << 20

# Metadata the output must not inherit from the reference GGUF.
DROPPED_KEYS = {"general.name", "general.file_type",
                "general.quantization_version", "general.alignment",
                "general.quantized_by", "general.url", "general.repo_url",
                "general.source.url", "general.source.repo_url"}


class Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        data = self.f.read(n)
        if len(data) != n:
            raise SystemExit("truncated GGUF")
        return data

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read(size))[0]

    def string(self):
        return self.read(self.unpack("Q"))


def raw_value(r, value_type):
    """The encoded bytes of one metadata value (without its type)."""
    if value_type == T_STRING:
        n = r.unpack("Q")
        return struct.pack("<Q", n) + r.read(n)
    if value_type == T_ARRAY:
        item_type = r.unpack("I")
        count = r.unpack("Q")
        parts = [struct.pack("<IQ", item_type, count)]
        if item_type == T_STRING:
            for _ in range(count):
                n = r.unpack("Q")
                parts.append(struct.pack("<Q", n) + r.read(n))
        else:
            parts.append(r.read(struct.calcsize("<" + SCALARS[item_type]) * count))
        return b"".join(parts)
    return r.read(struct.calcsize("<" + SCALARS[value_type]))


def read_reference(path):
    """Metadata entries (key -> (type, raw bytes)) and rope_freqs bytes."""
    with open(path, "rb") as f:
        r = Reader(f)
        if r.read(4) != GGUF_MAGIC:
            raise SystemExit(f"{path} is not a GGUF")
        r.unpack("I")
        tensor_count = r.unpack("Q")
        kv_count = r.unpack("Q")
        entries = {}
        alignment = ALIGNMENT
        for _ in range(kv_count):
            key = r.string().decode()
            value_type = r.unpack("I")
            raw = raw_value(r, value_type)
            if key == "general.alignment":
                alignment = struct.unpack("<I", raw)[0]
            entries[key] = (value_type, raw)
        rope = None
        for _ in range(tensor_count):
            name = r.string().decode()
            dims = [r.unpack("Q") for _ in range(r.unpack("I"))]
            ggml_type = r.unpack("I")
            offset = r.unpack("Q")
            if name == "rope_freqs.weight":
                rope = (dims, ggml_type, offset)
        data_start = -(-f.tell() // alignment) * alignment
        if rope is None or rope[1] != GGML_F32:
            raise SystemExit("reference GGUF has no F32 rope_freqs.weight")
        elements = 1
        for d in rope[0]:
            elements *= d
        f.seek(data_start + rope[2])
        rope_bytes = f.read(4 * elements)
    arch = entries.get("general.architecture")
    if arch is None or arch[1][8:] != b"diffusion-gemma":
        raise SystemExit("reference GGUF is not a diffusion-gemma model")
    for required in ("diffusion.canvas_length", "tokenizer.ggml.tokens",
                     "tokenizer.ggml.merges", "tokenizer.chat_template"):
        if required not in entries:
            raise SystemExit(f"reference GGUF lacks {required}")
    return entries, rope[0], rope_bytes


def read_checkpoint(root):
    """Decoder tensors: name -> (file, absolute offset, bytes, dtype, shape)."""
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = os.path.join(root, shard)
        with open(path, "rb") as f:
            header_bytes = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_bytes))
        for name, t in header.items():
            if name == "__metadata__":
                continue
            keep = (name.startswith("model.decoder.") or
                    name.startswith("model.encoder.language_model."))
            if not keep or name.endswith(".input_scale"):
                continue
            begin, end = t["data_offsets"]
            tensors[name] = (path, 8 + header_bytes + begin, end - begin,
                             t["dtype"], t["shape"])
    if not tensors:
        raise SystemExit("no decoder tensors in the checkpoint")
    for name, (_, _, size, dtype, shape) in tensors.items():
        if dtype not in FROM_SAFETENSORS:
            raise SystemExit(f"{name}: unsupported dtype {dtype}")
        e4m3 = name.endswith(".weight_scale")
        if (dtype == "F8_E4M3") != e4m3:
            raise SystemExit(f"{name}: breaks the *.weight_scale = E4M3 rule")
        elements = 1
        for d in shape:
            elements *= d
        if elements * FROM_SAFETENSORS[dtype][1] != size:
            raise SystemExit(f"{name}: shape {shape} does not match {size} bytes")
        if len(name.encode()) > 63:
            raise SystemExit(f"{name}: longer than a GGUF tensor name may be")
    return tensors


def pack_string(s):
    data = s.encode()
    return struct.pack("<Q", len(data)) + data


def kv_string(key, value):
    return pack_string(key) + struct.pack("<I", T_STRING) + pack_string(value)


def pad(n):
    return (-n) % ALIGNMENT


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-from", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-repo",
                        default="nvidia/diffusiongemma-26B-A4B-it-NVFP4")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--name",
                        default="DiffusionGemma 26B-A4B-it NVFP4 (NVIDIA modelopt)")
    args = parser.parse_args()

    entries, rope_dims, rope_bytes = read_reference(args.metadata_from)
    tensors = read_checkpoint(args.checkpoint)

    # Metadata: the reference's, minus what describes its own quantization.
    kv = [pack_string("general.alignment") + struct.pack("<II", T_UINT32, ALIGNMENT),
          kv_string("general.name", args.name),
          kv_string("diffusion_gemma.weights", WEIGHTS_FORMAT),
          kv_string("diffusion_gemma.source.repo", args.source_repo)]
    if args.source_revision:
        kv.append(kv_string("diffusion_gemma.source.revision", args.source_revision))
    for key, (value_type, raw) in entries.items():
        if key in DROPPED_KEYS:
            continue
        kv.append(pack_string(key) + struct.pack("<I", value_type) + raw)

    # Tensor infos: rope_freqs first, then the checkpoint's in name order.
    layout = [("rope_freqs.weight", None, len(rope_bytes), GGML_F32, rope_dims)]
    for name in sorted(tensors):
        path, offset, size, dtype, shape = tensors[name]
        ggml_type = FROM_SAFETENSORS[dtype][0]
        # GGUF lists dimensions innermost first.
        layout.append((name, (path, offset), size, ggml_type,
                       list(reversed(shape)) or [1]))
    infos, data_offset, offsets = [], 0, []
    for name, _, size, ggml_type, dims in layout:
        offsets.append(data_offset)
        infos.append(pack_string(name) + struct.pack("<I", len(dims)) +
                     b"".join(struct.pack("<Q", d) for d in dims) +
                     struct.pack("<IQ", ggml_type, data_offset))
        data_offset += size + pad(size)

    header = (GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(layout), len(kv)) +
              b"".join(kv) + b"".join(infos))
    header += b"\0" * pad(len(header))
    total = len(header) + data_offset
    print(f"{len(layout)} tensors, {total / 2**30:.2f} GiB -> {args.out}", flush=True)

    digests = {}
    tmp = args.out + ".partial"
    with open(tmp, "wb") as out:
        out.write(header)
        done = 0
        for (name, source, size, _, _), offset in zip(layout, offsets):
            digest = hashlib.sha256()
            if source is None:
                out.write(rope_bytes)
                digest.update(rope_bytes)
            else:
                path, start = source
                with open(path, "rb") as f:
                    f.seek(start)
                    left = size
                    while left:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            raise SystemExit(f"{name}: short read")
                        out.write(chunk)
                        digest.update(chunk)
                        left -= len(chunk)
            out.write(b"\0" * pad(size))
            digests[name] = digest.hexdigest()
            done += size
            if done // (1 << 30) != (done - size) // (1 << 30):
                print(f"  {done / 2**30:.1f} GiB", flush=True)

    # Read every tensor back from the written file.
    with open(tmp, "rb") as f:
        for (name, _, size, _, _), offset in zip(layout, offsets):
            f.seek(len(header) + offset)
            digest = hashlib.sha256()
            left = size
            while left:
                chunk = f.read(min(CHUNK, left))
                digest.update(chunk)
                left -= len(chunk)
            if digest.hexdigest() != digests[name]:
                raise SystemExit(f"{name}: written bytes differ from the source")
    os.replace(tmp, args.out)
    manifest = args.out + ".tensors.sha256"
    with open(manifest, "w") as f:
        for name, _, _, _, _ in layout:
            f.write(f"{digests[name]}  {name}\n")
    print(f"verified {len(layout)} tensors; per-tensor hashes in {manifest}")


if __name__ == "__main__":
    sys.exit(main())
