package com.xy.network.xor;

import com.xy.network.mnist.NeuralNetwork;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class XORDemo {

    public static void main(String[] args) throws IOException {
        Random random = new Random();

        Path file = Paths.get("target", "xor.network");
        NeuralNetwork network;
        if (Files.exists(file)) {
            network = new NeuralNetwork(file);
        } else {
            network = new NeuralNetwork(2, 5, 5, 1);
            network.randomize(random::nextDouble);
        }

        int[] values = {0, 1};
        for (int lap = 0; lap < 10_000; lap++) {
            int learning = 0;
            for (int x : values) {
                for (int y : values) {
                    RealVector input = new ArrayRealVector(new double[] {x, y});
                    RealVector target = new ArrayRealVector(new double[] {x ^ y});
                    learning += network.learn(input, target, 0.75, 100, 0.001);
                }
            }

            System.out.println("#" + lap + ": leaning: " + learning);
            for (int x : values) {
                for (int y : values) {
                    RealVector input = new ArrayRealVector(new double[] {x, y});
                    System.out.println(input + " --> " + network.decide(input));
                }
            }
            
            if (learning == 0) break;
        }

        network.write(file);
    }

}



