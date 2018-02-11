package com.xy.network.plus;

import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;

import com.xy.network.mnist.NeuralNetwork;
import java.util.Random;
import java.util.logging.Logger;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

public class PlusDemo {

    public static void main(String[] args) {
        Random random = new Random();
        Logger logger = Logger.getLogger("Plus/Minus");
        
        RealVector plus = MatrixUtils.createRealVector(new double[] { 0, 1, 0, 1, 1, 1, 0, 1, 0 });
        RealVector minus = MatrixUtils.createRealVector(new double[] { 0, 0, 0, 1, 1, 1, 0, 0, 0 });

        NeuralNetwork network = new NeuralNetwork(9, 18, 15, 1);
        network.randomize(random::nextDouble);
        for (int lap = 0; lap < 10_000; lap++) {
            RealVector isPlus = createRealVector(new double[] { 1 });
            RealVector isMinus = createRealVector(new double[] { 0 });
            int leanLaps = network.learn(plus, isPlus, 1, 100, 0.001)
                    + network.learn(minus, isMinus, 1, 100, 0.001);
            logger.info("lap #" + lap + ": " + leanLaps + " learning laps, results: " +
                    network.decide(plus) + ", " + network.decide(minus) + "");
            if (leanLaps == 0) break;
        }
    }

}
