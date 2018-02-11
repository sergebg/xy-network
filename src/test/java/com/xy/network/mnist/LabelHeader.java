package com.xy.network.mnist;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class LabelHeader {
    private static final int EXPECTED_MAGIC_NUMBER = 0x801;

    private final int size;

    public static LabelHeader read(FileChannel channel) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate(2 * Integer.BYTES);
        channel.read(bytes);
        if (bytes.hasRemaining()) {
            throw new IllegalArgumentException("Corrupted file header");
        }
        bytes.flip();
        int magicNumber = bytes.getInt();
        int size = bytes.getInt();
        if (EXPECTED_MAGIC_NUMBER != magicNumber) {
            throw new IllegalArgumentException("Expected magic number 0x" +
                    Integer.toHexString(EXPECTED_MAGIC_NUMBER) +
                    ", was 0x" + Integer.toHexString(magicNumber));
        }
        if (size < 0) {
            throw new IllegalArgumentException("Negative size " + size);
        }
        return new LabelHeader(size);
    }

    private LabelHeader(int size) {
        this.size = size;
    }

    public int getSize() {
        return size;
    }

}
