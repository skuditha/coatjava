package org.jlab.rec.alert.AIPID;

import java.util.Arrays;

/**
 * One fully resolved, strictly valid post-KF PID candidate.
 */
public final class PostPIDCandidate {

    public final int trackid;
    public final int clusterid;
    private final float[] features;

    public PostPIDCandidate(int trackid, int clusterid, float[] features) {
        if (features == null || features.length != PostPIDContract.FEATURE_COUNT) {
            throw new IllegalArgumentException(
                    "PostPID candidate requires float[" + PostPIDContract.FEATURE_COUNT + "]");
        }
        this.trackid = trackid;
        this.clusterid = clusterid;
        this.features = Arrays.copyOf(features, features.length);
    }

    /**
     * Returns a defensive copy of the raw engineered feature vector.
     * The TorchScript model performs standardization internally.
     */
    public float[] getFeatures() {
        return Arrays.copyOf(features, features.length);
    }
}
