package com.xy.network.mnist;

public class MNISTSample {
    private final int    label;
    private final byte[] image;

    public MNISTSample(int label, byte[] image) {
        this.label = label;
        this.image = image;
    }

    public int getLabel() {
        return label;
    }

    public byte[] getImage() {
        return image;
    }

    public StringBuilder print(StringBuilder sb, int width) {
        int i = 0;
        char[] pseudo = { ' ', ':', '*', '#' };
        while (i < image.length) {
            for (int j = 0; j < width; j++) {
                int value = image[i + j] & 0xFF;
                int index = value * 4 / 256;
                sb.append(pseudo[index]);
            }
            sb.append('\n');
            i += width;
        }
        return sb;
    }

}
