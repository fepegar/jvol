from __future__ import annotations

import numpy as np
from dahuffman import HuffmanCodec
from dahuffman.huffmancodec import _EndOfFileSymbol

from .casting import to_min_scalar_type
from .types import TypeRleCounts
from .types import TypeRleValues


class HuffmanCoding:
    def __init__(
        self,
        data: bytes,
        symbols_values: list[int],
        symbols_counts: list[int],
        bitsizes: list[int],
        values: list[int],
        eof_symbol: tuple[int, int],
    ):
        self.data = np.frombuffer(data, dtype=np.uint8)
        self.symbols_values = to_min_scalar_type(symbols_values)
        self.symbols_counts = to_min_scalar_type(symbols_counts)
        self.bitsizes = to_min_scalar_type(bitsizes)
        self.values = to_min_scalar_type(values)
        self.eof_symbol = to_min_scalar_type(eof_symbol)

    @classmethod
    def from_rle(
        cls,
        rle_values: TypeRleValues,
        rle_counts: TypeRleCounts,
    ) -> HuffmanCoding:
        sequence = list(zip(rle_values.tolist(), rle_counts.tolist()))
        codec = HuffmanCodec.from_data(sequence)
        symbol_to_code = codec.get_code_table()
        encoded_sequence = codec.encode(sequence)

        eof_symbol = symbol_to_code.pop(_EndOfFileSymbol())
        symbols_values, symbols_counts, bitsizes, values = cls._unpack_huffman_table(
            symbol_to_code,
        )
        huffman_coding = cls(
            data=encoded_sequence,
            symbols_values=symbols_values,
            symbols_counts=symbols_counts,
            bitsizes=bitsizes,
            values=values,
            eof_symbol=eof_symbol,
        )
        return huffman_coding

    @staticmethod
    def _unpack_huffman_table(table):
        symbols_values, symbols_counts, bitsizes, values = zip(
            *((sv, sc, bs, val) for (sv, sc), (bs, val) in table.items())
        )
        return symbols_values, symbols_counts, bitsizes, values

    def decode(self) -> tuple[TypeRleValues, TypeRleCounts]:
        table = dict(
            zip(
                zip(self.symbols_values.tolist(), self.symbols_counts.tolist()),
                zip(self.bitsizes.tolist(), self.values.tolist()),
            ),
        )
        # table[_EndOfFileSymbol()] = self.eof_symbol
        eof_tuple = tuple(self.eof_symbol.tolist())
        # table[eof_tuple] = _EndOfFileSymbol()
        table[_EndOfFileSymbol()] = eof_tuple
        codec = HuffmanCodec(table)
        sequence = codec.decode(self.data)
        rle_values, rle_counts = zip(*sequence)
        return rle_values, rle_counts
