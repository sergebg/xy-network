package com.xy.network.mnist;

import static java.util.logging.Logger.getLogger;

import com.xy.network.mnist.MNISTSample;
import com.xy.network.mnist.MNISTSamples;
import com.xy.network.mnist.NeuralNetwork;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.SystemUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

class MNISTDemo {
    private static final String TEST_IMAGE_LABEL_URL  = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGE_URL        = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
    private static final String TRAIN_IMAGE_LABEL_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
    private static final String TRAIN_IMAGE_URL       = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";

    private static final int TEST_LIMIT  = 10_000;
    private static final int TRAIN_LIMIT = 60_000;

    private static final Logger logger = getLogger(MNISTDemo.class.getName());

    public static void main(String[] args) throws IOException {
        Path trainImagePath = download(TRAIN_IMAGE_URL);
        Path trainLabelPath = download(TRAIN_IMAGE_LABEL_URL);
        Path testImagePath = download(TEST_IMAGE_URL);
        Path testLabelPath = download(TEST_IMAGE_LABEL_URL);

        List<MNISTSample> trainSamples = MNISTSamples.stream(trainLabelPath, trainImagePath)
                .limit(TRAIN_LIMIT)
                .collect(Collectors.toList());

        List<MNISTSample> testSamples = MNISTSamples.stream(testLabelPath, testImagePath)
                .limit(TEST_LIMIT)
                .collect(Collectors.toList());

        logger.info("Initializing network");
        Path file = Paths.get("target", "mnist.network");
        NeuralNetwork network;
        if (Files.exists(file)) {
            network = new NeuralNetwork(file);
            logger.info("Loaded from file");
        } else {
            network = new NeuralNetwork(
                    28 * 28,
                    450,
                    250,
                    10);
            Random random = new Random();
            network.randomize(random::nextDouble);
            logger.info("Random");
        }

        double learnRate = 0.025;
        int maxIterations = 25;
        double epsilon = 0.01;

        for (int lap = 0; lap < 25; lap++) {

            int learning = 0;
            for (MNISTSample sample : trainSamples) {
                RealVector input = toInputVector(sample);
                RealVector target = new ArrayRealVector(10);
                target.setEntry(sample.getLabel(), 1);
                learning += network.learn(input, target, learnRate, maxIterations, epsilon);
            }

            int trainLap = ++lap;
            int trainErrors = test(network, trainSamples, false);
            int testErrors = test(network, testSamples, false);
            logger.info(String.format("lap #%03d  %3d  %3d  %d", trainLap, trainErrors, testErrors, learning));

            if (learning == 0) {
                break;
            }
        }

        network.write(file);
    }

    private static int test(NeuralNetwork network, List<MNISTSample> subList, boolean print) {
        int errors = 0;
        for (MNISTSample sample : subList) {
            RealVector input = toInputVector(sample);
            RealVector output = network.decide(input);
            if (sample.getLabel() != output.getMaxIndex()) {
                errors++;
                if (print) {
                    logger.info(() -> sample.getLabel() + ": " + output + "\n" + sample.print(new StringBuilder(), 28));
                }
            }
        }
        return errors;
    }

    private static RealVector toInputVector(MNISTSample sample) {
        byte[] image = sample.getImage();
        RealVector input = new ArrayRealVector(image.length);
        for (int i = 0; i < image.length; i++) {
            input.setEntry(i, (0xFF & image[i]) / 200.0);
        }
        return input;
    }

    private static Path download(String urlString) throws IOException {
        URL url = new URL(urlString);
        String urlPath = extractPath(url);
        Path tmpDir = SystemUtils.getUserHome().toPath();
        Path file = tmpDir.resolve(urlPath).normalize();
        if (!Files.exists(file)) {
            logger.info(() -> "Downloading " + file);
            Files.createDirectories(file.getParent());
            try (GZIPInputStream inputStream = new GZIPInputStream(url.openStream());
                    FileOutputStream outputStream = new FileOutputStream(file.toFile())) {
                IOUtils.copy(inputStream, outputStream);
            }
        } else {
            logger.info(() -> "Reading existing file " + file);
        }
        return file;
    }

    private static String extractPath(URL url) {
        String path = url.getFile();
        if (!path.startsWith("/")) throw new IllegalStateException();
        path = path.substring(1);
        if (!path.endsWith(".gz")) throw new IllegalStateException();
        return path.substring(0, path.length() - 3);
    }

}
