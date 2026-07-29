package org.jlab.rec.alert.AIPID;

/**
 * Frozen deployment contract for the ALERT post-KF PID v2 model.
 *
 * <p>The Java feature builder, TorchScript model, and output-bank mapping must
 * all use this ordering. Changing any value here requires a new model contract
 * and a retrained/exported model.</p>
 */
public final class PostPIDContract {

    public static final int FEATURE_COUNT = 18;
    public static final int CLASS_COUNT = 5;

    public static final int KF_X = 0;
    public static final int KF_Y = 1;
    public static final int KF_Z = 2;
    public static final int LOG1P_P = 3;
    public static final int LOG1P_PT = 4;
    public static final int PHI = 5;
    public static final int THETA = 6;
    public static final int N_HITS = 7;
    public static final int N_BAR = 8;
    public static final int N_WEDGE = 9;
    public static final int LOG1P_DEDX = 10;
    public static final int SUM_RESIDUALS = 11;
    public static final int LOG1P_ATOF_ENERGY = 12;
    public static final int ATOF_X = 13;
    public static final int ATOF_Y = 14;
    public static final int ATOF_Z = 15;
    public static final int TOF = 16;
    public static final int PATHLENGTH = 17;

    private static final int[] CLASS_PIDS = {2212, 45, 46, 49, 47};

    private PostPIDContract() {
        // Utility class.
    }

    /**
     * Returns the LUND PID corresponding to a model output index.
     *
     * @param classIndex model output index in [0, 4]
     * @return LUND PID
     */
    public static int pidForClassIndex(int classIndex) {
        if (classIndex < 0 || classIndex >= CLASS_PIDS.length) {
            throw new IllegalArgumentException("Invalid PostPID class index: " + classIndex);
        }
        return CLASS_PIDS[classIndex];
    }
}
