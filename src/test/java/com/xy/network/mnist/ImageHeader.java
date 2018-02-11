package com.xy.network.mnist;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class ImageHeader {
    private static final int EXPECTED_MAGIC_NUMBER = 0x803;

    private final int size;
    private final int height;
    private final int width;

    public static ImageHeader read(FileChannel channel) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate(Integer.BYTES * 4);
        channel.read(bytes);
        if (bytes.hasRemaining()) {
            throw new IllegalArgumentException("Corrupted file header");
        }
        bytes.flip();
        int magicNumber = bytes.getInt();
        if (EXPECTED_MAGIC_NUMBER != magicNumber) {
            throw new IllegalArgumentException("Expected magic number 0x" +
                    Integer.toHexString(EXPECTED_MAGIC_NUMBER) +
                    ", was 0x" + Integer.toHexString(magicNumber));
        }
        int size = bytes.getInt();
        if (size < 0) {
            throw new IllegalArgumentException("Illegal size " + size);
        }
        int height = bytes.getInt();
        if (height < 1) {
            throw new IllegalArgumentException("Illegal height " + height);
        }
        int width = bytes.getInt();
        if (width < 1) {
            throw new IllegalArgumentException("Illegal width " + width);
        }
        return new ImageHeader(size, height, width);
    }

    private ImageHeader(int size, int height, int width) {
        this.size = size;
        this.height = height;
        this.width = width;
    }

    public int getSize() {
        return size;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

}
