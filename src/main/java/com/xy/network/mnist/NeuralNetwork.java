package com.xy.network.mnist;

import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.function.DoubleSupplier;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class NeuralNetwork {
    private final UnivariateFunction sigmoid = new Sigmoid(0, 1);
    private final List<RealMatrix>   weight;

    public NeuralNetwork(int... dimension) {
        if (dimension.length < 2) {
            throw new IllegalArgumentException("Expected > 1, was " + dimension.length);
        }
        List<RealMatrix> weightBuilder = new LinkedList<>();
        for (int i = 1; i < dimension.length; i++) {
            weightBuilder.add(MatrixUtils.createRealMatrix(dimension[i - 1], dimension[i]));
        }
        weight = Collections.unmodifiableList(new ArrayList<>(weightBuilder));
    }

    public NeuralNetwork(Path file) throws IOException {
        List<RealMatrix> matrixList = new LinkedList<>();
        try (FileChannel channel = FileChannel.open(file, READ)) {

            ByteBuffer bytes = ByteBuffer.allocate(16 * 1024);
            bytes.flip();

            int size = getInt(channel, bytes);
            for (int i = 0; i < size; i++) {
                int rows = getInt(channel, bytes);
                int cols = getInt(channel, bytes);
                RealMatrix matrix = MatrixUtils.createRealMatrix(rows, cols);
                matrix.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {

                    @Override
                    public double visit(int row, int column, double value) {
                        return getDouble(channel, bytes);
                    }

                });
                matrixList.add(matrix);
            }
        }
        weight = Collections.unmodifiableList(new ArrayList<>(matrixList));
    }

    public void randomize(DoubleSupplier supplier) {
        for (RealMatrix matrix : weight) {
            for (int i = 0; i < matrix.getRowDimension(); i++) {
                for (int j = 0; j < matrix.getColumnDimension(); j++) {
                    matrix.setEntry(i, j, supplier.getAsDouble());
                }
            }
            double factor = 1 / matrix.getFrobeniusNorm();
            matrix.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {

                @Override
                public double visit(int row, int column, double value) {
                    return value * factor;
                }

            });
        }
    }

    public int learn(RealVector input, RealVector target, double rate, int maxIterations, double epsilon) {
        int size = weight.size();
        for (int lap = 0; lap < maxIterations; lap++) {
            List<RealVector> o = new ArrayList<>(size + 1);

            RealVector output = input;
            o.add(output);
            for (int i = 0; i < size; i++) {
                o.add(output = weight.get(i).preMultiply(output).map(sigmoid));
            }

            RealVector error = target.subtract(output);
            if (Arrays.stream(error.toArray()).allMatch(v -> Math.abs(v) < epsilon)) {
                return lap;
            }

            List<RealVector> e = new ArrayList<>(size);
            e.add(error);
            for (int i = size - 1; i >= 1; i--) {
                e.add(error = weight.get(i).transpose().preMultiply(error));
            }
            Collections.reverse(e);

            for (int i = size - 1; i >= 0; i--) {
                RealMatrix delta = o.get(i)
                        .outerProduct(e.get(i).ebeMultiply(o.get(i + 1)).ebeMultiply(o.get(i + 1).mapSubtract(1)));
                weight.get(i).setSubMatrix(weight.get(i).subtract(delta.scalarMultiply(rate)).getData(), 0, 0);
            }
        }
        return maxIterations;
    }

    public RealVector decide(RealVector input) {
        for (int i = 0; i < weight.size(); i++) {
            input = weight.get(i).preMultiply(input).map(sigmoid);
        }
        return input;
    }

    public void write(Path file) throws IOException {
        try (FileChannel output = FileChannel.open(file, CREATE, TRUNCATE_EXISTING, WRITE)) {
            int maxSize = weight.stream().mapToInt(m -> m.getRowDimension() * m.getColumnDimension() * Double.BYTES +
                    3 * Integer.BYTES).max().getAsInt();
            ByteBuffer bytes = ByteBuffer.allocate(maxSize);
            bytes.putInt(weight.size());

            for (int i = 0; i < weight.size(); i++) {
                RealMatrix w = weight.get(i);
                bytes.putInt(w.getRowDimension());
                bytes.putInt(w.getColumnDimension());
                w.walkInRowOrder(new DefaultRealMatrixPreservingVisitor() {

                    @Override
                    public void visit(int row, int column, double value) {
                        bytes.putDouble(value);
                    }

                });

                bytes.flip();
                while (bytes.hasRemaining()) {
                    output.write(bytes);
                }
                bytes.clear();
            }
        }
    }

    private static int getInt(FileChannel channel, ByteBuffer bytes) {
        readMoreBytes(channel, bytes, Integer.BYTES);
        return bytes.getInt();
    }

    private static double getDouble(FileChannel channel, ByteBuffer bytes) {
        readMoreBytes(channel, bytes, Double.BYTES);
        return bytes.getDouble();
    }

    private static void readMoreBytes(FileChannel channel, ByteBuffer bytes, int size) {
        if (bytes.remaining() < size) {
            bytes.compact();
            try {
                readBytes(channel, bytes);
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
            bytes.flip();
        }
    }

    private static void readBytes(FileChannel channel, ByteBuffer bytes) throws IOException {
        while (bytes.hasRemaining()) {
            int n = channel.read(bytes);
            if (n == -1) {
                break;
            }
        }
    }

}
