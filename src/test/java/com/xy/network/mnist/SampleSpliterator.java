package com.xy.network.mnist;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Spliterator;
import java.util.function.Consumer;

class SampleSpliterator implements Spliterator<MNISTSample> {

    private static final int IMAGE_BUFFER_SIZE = 256 * 1024;
    private static final int LABEL_BUFFER_SIZE = 16 * 1024;
    private static final int CHARACTERISTICS   = ORDERED | IMMUTABLE | SIZED | SUBSIZED;

    private final FileChannel imageChannel;
    private final FileChannel labelChannel;
    private final ByteBuffer  imageBytes;
    private final ByteBuffer  labelBytes;

    private final int size;
    private final int imageSize;

    private int index;

    SampleSpliterator(Path imagePath, Path labelPath) throws IOException {
        imageChannel = FileChannel.open(imagePath, StandardOpenOption.READ);

        ImageHeader imageHeader = ImageHeader.read(imageChannel);
        imageSize = imageHeader.getHeight() * imageHeader.getWidth();
        size = imageHeader.getSize();

        labelChannel = FileChannel.open(labelPath, StandardOpenOption.READ);

        if (LabelHeader.read(labelChannel).getSize() != size) {
            throw new IllegalArgumentException("Inconsistent files");
        }

        imageBytes = ByteBuffer.allocate((IMAGE_BUFFER_SIZE + imageSize - 1) / imageSize * imageSize);
        imageBytes.flip();

        labelBytes = ByteBuffer.allocate(LABEL_BUFFER_SIZE);
        labelBytes.flip();
    }

    @Override
    public Spliterator<MNISTSample> trySplit() {
        return null; // reject split
    }

    @Override
    public long estimateSize() {
        return (size - index);
    }

    @Override
    public int characteristics() {
        return CHARACTERISTICS;
    }

    @Override
    public boolean tryAdvance(Consumer<? super MNISTSample> action) {
        if (index >= size) {
            return false;
        }

        if (!labelBytes.hasRemaining()) {
            readFileChannel(labelChannel, labelBytes);
        }

        if (!imageBytes.hasRemaining()) {
            readFileChannel(imageChannel, imageBytes);
        }

        index++;
        byte label = labelBytes.get();
        byte[] image = new byte[imageSize];
        imageBytes.get(image);
        action.accept(new MNISTSample(label, image));
        return true;
    }
    
    void close() {
        try (FileChannel ch1 = imageChannel; FileChannel ch2 = labelChannel) {
            //
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    private static void readFileChannel(FileChannel fileChannel, ByteBuffer byteBuffer) {
        byteBuffer.clear();
        try {
            while (byteBuffer.hasRemaining()) {
                if (-1 == fileChannel.read(byteBuffer)) {
                    fileChannel.close();
                    break;
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        byteBuffer.flip();
    }

}