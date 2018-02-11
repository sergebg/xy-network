package com.xy.network.mnist;

import java.io.IOException;
import java.nio.file.Path;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class MNISTSamples {

    public static Stream<MNISTSample> stream(Path labelPath, Path imagePath) throws IOException {
        SampleSpliterator spliterator = new SampleSpliterator(imagePath, labelPath);
        return StreamSupport.stream(spliterator, false).onClose(spliterator::close);
    }

    private MNISTSamples() {
        //
    }

}
