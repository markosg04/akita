#!/usr/bin/env python3
"""Independent hashlib vectors for the epoch-6 stream and setup framing."""
import hashlib
import json
import struct


def block(domain, context, index):
    message = (struct.pack('<I', len(domain)) + domain +
               struct.pack('<Q', len(context)) + context + struct.pack('<Q', index))
    return hashlib.blake2b(message, digest_size=64).digest()


def setup_samples(bits, offset):
    modulus = 2**bits - offset
    domain = b'akita/public-matrix/blake2b512-paged/v2'
    result = []
    for page in (0, 1):
        context = bytes([31])*32 + modulus.to_bytes(32, 'big') + struct.pack('<QQ', 4096, page)
        values = []
        index = 0
        while len(values) < 4096:
            chunk = block(domain, context, index)
            index += 1
            for start in range(0, len(chunk), bits//8):
                value = int.from_bytes(chunk[start:start+bits//8], 'little')
                if value < modulus:
                    values.append(value)
        result.extend([values[0], values[4095]] if page == 0 else [values[0]])
    return result


def a7f7_samples():
    modulus = 2**128 - 2**32 + 22537
    context = bytes([71])*32 + modulus.to_bytes(32, 'big') + struct.pack('<QQ', 4096, 0)
    domain = b'akita/public-matrix/blake2b512-paged/v2'
    raw = b''.join(block(domain, context, i) for i in range(4))
    values, consumed = [], 0
    while len(values) < 8:
        candidate = int.from_bytes(raw[consumed:consumed+16], 'little')
        consumed += 16
        if candidate < modulus:
            values.append(candidate)
    return dict(values=values, consumed=consumed, suffix=raw[consumed:consumed+16].hex())


def bn254_samples():
    modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    context = bytes([71])*32 + modulus.to_bytes(32, 'big') + struct.pack('<QQ', 4096, 0)
    domain = b'akita/public-matrix/blake2b512-paged/v2'
    raw = b''.join(block(domain, context, i) for i in range(8))
    values, consumed = [], 0
    while len(values) < 8:
        candidate = int.from_bytes(raw[consumed:consumed+32], 'little') & ((1 << 254)-1)
        consumed += 32
        if candidate < modulus:
            values.append(candidate.to_bytes(32, 'little').hex())
    return dict(values=values, consumed=consumed, suffix=raw[consumed:consumed+32].hex())


def transcript_schedule():
    h = lambda message: hashlib.blake2b(message, digest_size=64).digest()
    label, instance = b'ak/p', b'akita/default-instance'
    message = (b'akita-pcs/transcript/v2/blake2b'.ljust(64, b'\0') +
               b'akita-pcs/session-label/v1'.ljust(64, b'\0') +
               struct.pack('<Q', len(label)) + label + struct.pack('<Q', len(instance)) + instance)
    cv = bytes(64)
    values = []
    framed = lambda value: struct.pack('<Q', len(value)) + value
    for absorbed in (message + framed(b'C') + framed(b'O'), framed(b'RS'), framed(b'SC1')):
        cv = h(h(bytes(128) + cv + absorbed))
        challenge = h(bytes(127) + b'\1' + cv + bytes(8))[:16]
        values.append(f'{int.from_bytes(challenge, "little") % (2**64-59):016x}')
        cv = h(bytes(127) + b'\2' + cv + (32).to_bytes(8, 'big'))
    return values


def transcript_challenge():
    h = lambda message: hashlib.blake2b(message, digest_size=64).digest()
    label, instance = b'backend-test', b'instance'
    message = (b'akita-pcs/transcript/v2/blake2b'.ljust(64, b'\0') +
               b'akita-pcs/session-label/v1'.ljust(64, b'\0') +
               struct.pack('<Q', len(label)) + label + struct.pack('<Q', len(instance)) + instance)
    cv = h(h(bytes(192) + message))
    challenge = h(bytes(127) + b'\1' + cv + bytes(8))[:32]
    return int.from_bytes(challenge, 'little') % (2**128 - 275)


if __name__ == '__main__':
    sparse = b'akita/sparse-challenge/blake2b512/v1'
    matrix_context = bytes(32) + struct.pack('<Q', 1) + b'A' + struct.pack('<QQQQ', 2, 2, 0, 0)
    print(json.dumps({
        'sparse_zero_blocks': [block(sparse, bytes(40), j).hex() for j in (0, 1)],
        'sparse_indexed': block(sparse, bytes(range(32)) + struct.pack('<Q', 0x0102030405060708), 0).hex(),
        'matrix_entry': block(b'akita/matrix-entry/blake2b512/v1', matrix_context, 0).hex(),
        'setup': {str(bits): setup_samples(bits, offset) for bits, offset in ((32, 99), (64, 59), (128, 275))},
        'transcript': transcript_challenge(),
        'transcript_schedule': transcript_schedule(),
        'bn254_canonical': bn254_samples(),
        'a7f7_boundary': setup_samples(128, 2**32 - 22537),
        'a7f7_consumption': a7f7_samples(),
    }, indent=2))
