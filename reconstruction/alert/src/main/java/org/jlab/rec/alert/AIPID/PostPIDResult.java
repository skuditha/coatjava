package org.jlab.rec.alert.AIPID;

/**
 * Post-KF PID inference result in output-bank field order.
 */
public final class PostPIDResult {

    public final int trackid;
    public final int clusterid;
    public final int pid;
    public final float prob2212;
    public final float prob45;
    public final float prob46;
    public final float prob47;
    public final float prob49;

    public PostPIDResult(
            int trackid,
            int clusterid,
            int pid,
            float prob2212,
            float prob45,
            float prob46,
            float prob47,
            float prob49) {
        this.trackid = trackid;
        this.clusterid = clusterid;
        this.pid = pid;
        this.prob2212 = prob2212;
        this.prob45 = prob45;
        this.prob46 = prob46;
        this.prob47 = prob47;
        this.prob49 = prob49;
    }
}
