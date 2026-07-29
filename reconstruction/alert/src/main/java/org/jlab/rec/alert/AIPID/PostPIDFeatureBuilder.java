package org.jlab.rec.alert.AIPID;

import org.jlab.io.base.DataBank;
import org.jlab.io.base.DataEvent;

/**
 * Resolves the post-KF AHDC/ATOF association and constructs the frozen v2
 * 18-feature vector.
 *
 * <p>This class performs no model inference and no normalization. The exported
 * TorchScript model contains the fitted standardization constants.</p>
 */
public final class PostPIDFeatureBuilder {

    private static final int NO_MATCH = -1;
    private static final int DUPLICATE_MATCH = -2;

    /** Reason a candidate was not produced. */
    public enum Status {
        SUCCESS,
        MISSING_REQUIRED_BANK,
        INVALID_PROJECTION_ROW_COUNT,
        INVALID_TRACK_ID,
        INVALID_MATCHED_HIT_ID,
        MISSING_KFTRACK_MATCH,
        DUPLICATE_KFTRACK_MATCH,
        MISSING_ATOF_HIT_MATCH,
        DUPLICATE_ATOF_HIT_MATCH,
        INVALID_CLUSTER_ID,
        MISSING_ATOF_CLUSTER_MATCH,
        DUPLICATE_ATOF_CLUSTER_MATCH,
        INVALID_START_TIME,
        INVALID_COUNT,
        NONFINITE_INPUT,
        NEGATIVE_DEDX,
        NEGATIVE_ATOF_ENERGY,
        NONPOSITIVE_MOMENTUM,
        NONPOSITIVE_PATHLENGTH,
        FLOAT_OVERFLOW
    }

    /** Result wrapper that preserves the rejection reason for diagnostics. */
    public static final class BuildResult {
        private final Status status;
        private final PostPIDCandidate candidate;

        private BuildResult(Status status, PostPIDCandidate candidate) {
            this.status = status;
            this.candidate = candidate;
        }

        public static BuildResult success(PostPIDCandidate candidate) {
            return new BuildResult(Status.SUCCESS, candidate);
        }

        public static BuildResult rejected(Status status) {
            if (status == Status.SUCCESS) {
                throw new IllegalArgumentException("SUCCESS requires a candidate");
            }
            return new BuildResult(status, null);
        }

        public boolean isSuccess() {
            return status == Status.SUCCESS;
        }

        public Status getStatus() {
            return status;
        }

        public PostPIDCandidate getCandidate() {
            return candidate;
        }
    }

    /**
     * Builds the single candidate for an event.
     *
     * <p>Exactly one ALERT::ai:projections row is required. All bank links must
     * resolve uniquely. Feature-level invalidity is handled strictly: no masks
     * or default values are used. Negative corrected TOF is accepted.</p>
     */
    public BuildResult build(DataEvent event) {
        if (!hasRequiredBanks(event)) {
            return BuildResult.rejected(Status.MISSING_REQUIRED_BANK);
        }

        DataBank projectionBank = event.getBank("ALERT::ai:projections");
        if (projectionBank.rows() != 1) {
            return BuildResult.rejected(Status.INVALID_PROJECTION_ROW_COUNT);
        }

        int trackId = projectionBank.getInt("trackid", 0);
        if (trackId == -1) {
            return BuildResult.rejected(Status.INVALID_TRACK_ID);
        }

        int matchedHitId = projectionBank.getInt("matched_atof_hit_id", 0);
        if (matchedHitId == -1) {
            return BuildResult.rejected(Status.INVALID_MATCHED_HIT_ID);
        }

        DataBank kfTrackBank = event.getBank("AHDC::kftrack");
        int trackRow = findUniqueIntRow(kfTrackBank, "trackid", trackId);
        if (trackRow == DUPLICATE_MATCH) {
            return BuildResult.rejected(Status.DUPLICATE_KFTRACK_MATCH);
        }
        if (trackRow == NO_MATCH) {
            return BuildResult.rejected(Status.MISSING_KFTRACK_MATCH);
        }

        DataBank hitBank = event.getBank("ATOF::hits");
        int matchedHitRow = findUniqueShortRow(hitBank, "id", matchedHitId);
        if (matchedHitRow == DUPLICATE_MATCH) {
            return BuildResult.rejected(Status.DUPLICATE_ATOF_HIT_MATCH);
        }
        if (matchedHitRow == NO_MATCH) {
            return BuildResult.rejected(Status.MISSING_ATOF_HIT_MATCH);
        }

        int clusterId = hitBank.getShort("clusterid", matchedHitRow);
        if (clusterId == -1) {
            return BuildResult.rejected(Status.INVALID_CLUSTER_ID);
        }

        DataBank clusterBank = event.getBank("ATOF::clusters");
        int clusterRow = findUniqueShortRow(clusterBank, "id", clusterId);
        if (clusterRow == DUPLICATE_MATCH) {
            return BuildResult.rejected(Status.DUPLICATE_ATOF_CLUSTER_MATCH);
        }
        if (clusterRow == NO_MATCH) {
            return BuildResult.rejected(Status.MISSING_ATOF_CLUSTER_MATCH);
        }

        StartTimeResult startTimeResult = readStartTime(event);
        if (!startTimeResult.valid) {
            return BuildResult.rejected(Status.INVALID_START_TIME);
        }

        return buildFeatures(
                trackId,
                clusterId,
                trackRow,
                clusterRow,
                startTimeResult.value,
                kfTrackBank,
                hitBank,
                clusterBank);
    }

    private static BuildResult buildFeatures(
            int trackId,
            int clusterId,
            int trackRow,
            int clusterRow,
            double startTime,
            DataBank trackBank,
            DataBank hitBank,
            DataBank clusterBank) {

        double kfX = trackBank.getFloat("x", trackRow);
        double kfY = trackBank.getFloat("y", trackRow);
        double kfZ = trackBank.getFloat("z", trackRow);
        double px = trackBank.getFloat("px", trackRow);
        double py = trackBank.getFloat("py", trackRow);
        double pz = trackBank.getFloat("pz", trackRow);
        int nHits = trackBank.getInt("n_hits", trackRow);
        double dedx = trackBank.getFloat("dEdx", trackRow);
        double sumResiduals = trackBank.getFloat("sum_residuals", trackRow);

        int nBar = clusterBank.getInt("n_bar", clusterRow);
        int nWedge = clusterBank.getInt("n_wedge", clusterRow);
        double clusterTime = clusterBank.getFloat("time", clusterRow);
        double clusterX = clusterBank.getFloat("x", clusterRow);
        double clusterY = clusterBank.getFloat("y", clusterRow);
        double clusterZ = clusterBank.getFloat("z", clusterRow);
        double clusterEnergy = clusterBank.getFloat("energy", clusterRow);

        if (!allFinite(
                startTime,
                kfX, kfY, kfZ,
                px, py, pz,
                dedx, sumResiduals,
                clusterTime, clusterX, clusterY, clusterZ, clusterEnergy)) {
            return BuildResult.rejected(Status.NONFINITE_INPUT);
        }
        if (nHits <= 0 || nBar < 0 || nWedge < 0) {
            return BuildResult.rejected(Status.INVALID_COUNT);
        }
        if (dedx < 0.0) {
            return BuildResult.rejected(Status.NEGATIVE_DEDX);
        }
        if (clusterEnergy < 0.0) {
            return BuildResult.rejected(Status.NEGATIVE_ATOF_ENERGY);
        }

        double atofX = clusterX;
        double atofY = clusterY;
        double atofZ = clusterZ;
        double atofTime = clusterTime;

        int barHitRow = selectBarHitRow(hitBank, clusterId);
        if (barHitRow >= 0) {
            double barTime = hitBank.getFloat("time", barHitRow);
            double barX = hitBank.getFloat("x", barHitRow);
            double barY = hitBank.getFloat("y", barHitRow);
            double barZ = hitBank.getFloat("z", barHitRow);
            double barEnergy = hitBank.getFloat("energy", barHitRow);

            if (!allFinite(barTime, barX, barY, barZ, barEnergy) || barEnergy < 0.0) {
                return BuildResult.rejected(Status.NONFINITE_INPUT);
            }
            atofX = barX;
            atofY = barY;
            atofZ = barZ;
            atofTime = barTime;
        }

        double pt = Math.hypot(px, py);
        double p = Math.hypot(pt, pz);
        if (!Double.isFinite(p) || p <= 0.0) {
            return BuildResult.rejected(Status.NONPOSITIVE_MOMENTUM);
        }

        double cosTheta = clamp(pz / p, -1.0, 1.0);
        double theta = Math.acos(cosTheta);
        double phi = Math.atan2(py, px);
        double correctedTof = atofTime - startTime;

        double dx = atofX - kfX;
        double dy = atofY - kfY;
        double dz = atofZ - kfZ;
        double pathlength = Math.hypot(Math.hypot(dx, dy), dz);
        if (!Double.isFinite(pathlength) || pathlength <= 0.0) {
            return BuildResult.rejected(Status.NONPOSITIVE_PATHLENGTH);
        }

        if (!allFinite(pt, theta, phi, correctedTof, atofX, atofY, atofZ)) {
            return BuildResult.rejected(Status.NONFINITE_INPUT);
        }

        double[] values = new double[PostPIDContract.FEATURE_COUNT];
        values[PostPIDContract.KF_X] = kfX;
        values[PostPIDContract.KF_Y] = kfY;
        values[PostPIDContract.KF_Z] = kfZ;
        values[PostPIDContract.LOG1P_P] = Math.log1p(p);
        values[PostPIDContract.LOG1P_PT] = Math.log1p(pt);
        values[PostPIDContract.PHI] = phi;
        values[PostPIDContract.THETA] = theta;
        values[PostPIDContract.N_HITS] = nHits;
        values[PostPIDContract.N_BAR] = nBar;
        values[PostPIDContract.N_WEDGE] = nWedge;
        values[PostPIDContract.LOG1P_DEDX] = Math.log1p(dedx);
        values[PostPIDContract.SUM_RESIDUALS] = sumResiduals;
        values[PostPIDContract.LOG1P_ATOF_ENERGY] = Math.log1p(clusterEnergy);
        values[PostPIDContract.ATOF_X] = atofX;
        values[PostPIDContract.ATOF_Y] = atofY;
        values[PostPIDContract.ATOF_Z] = atofZ;
        values[PostPIDContract.TOF] = correctedTof;
        values[PostPIDContract.PATHLENGTH] = pathlength;

        float[] features = new float[PostPIDContract.FEATURE_COUNT];
        for (int index = 0; index < values.length; index++) {
            if (!fitsFloat(values[index])) {
                return BuildResult.rejected(Status.FLOAT_OVERFLOW);
            }
            features[index] = (float) values[index];
        }

        return BuildResult.success(new PostPIDCandidate(trackId, clusterId, features));
    }

    private static boolean hasRequiredBanks(DataEvent event) {
        return event.hasBank("ALERT::ai:projections")
                && event.hasBank("AHDC::kftrack")
                && event.hasBank("ATOF::hits")
                && event.hasBank("ATOF::clusters")
                && (event.hasBank("MC::Particle") || event.hasBank("REC::Event"));
    }

    private static int findUniqueIntRow(DataBank bank, String field, int target) {
        int match = NO_MATCH;
        for (int row = 0; row < bank.rows(); row++) {
            if (bank.getInt(field, row) != target) {
                continue;
            }
            if (match != NO_MATCH) {
                return DUPLICATE_MATCH;
            }
            match = row;
        }
        return match;
    }

    private static int findUniqueShortRow(DataBank bank, String field, int target) {
        int match = NO_MATCH;
        for (int row = 0; row < bank.rows(); row++) {
            if (bank.getShort(field, row) != target) {
                continue;
            }
            if (match != NO_MATCH) {
                return DUPLICATE_MATCH;
            }
            match = row;
        }
        return match;
    }

    /**
     * Selects the component-10 hit in the resolved cluster with the highest
     * energy. Equal energies are broken by the lowest hit ID, then row index.
     */
    private static int selectBarHitRow(DataBank hitBank, int clusterId) {
        int bestRow = NO_MATCH;
        int bestId = Integer.MAX_VALUE;
        double bestEnergy = 0.0;

        for (int row = 0; row < hitBank.rows(); row++) {
            if (hitBank.getShort("clusterid", row) != clusterId
                    || hitBank.getInt("component", row) != 10) {
                continue;
            }

            int id = hitBank.getShort("id", row);
            double energy = hitBank.getFloat("energy", row);
            if (bestRow == NO_MATCH
                    || energy > bestEnergy
                    || (energy == bestEnergy && id < bestId)
                    || (energy == bestEnergy && id == bestId && row < bestRow)) {
                bestRow = row;
                bestId = id;
                bestEnergy = energy;
            }
        }
        return bestRow;
    }

    private static StartTimeResult readStartTime(DataEvent event) {
        if (event.hasBank("MC::Particle")) {
            DataBank mcBank = event.getBank("MC::Particle");
            if (mcBank.rows() != 1) {
                return StartTimeResult.invalid();
            }
            double value = mcBank.getFloat("vt", 0);
            return Double.isFinite(value)
                    ? StartTimeResult.valid(value)
                    : StartTimeResult.invalid();
        }

        if (!event.hasBank("REC::Event")) {
            return StartTimeResult.invalid();
        }
        DataBank recEventBank = event.getBank("REC::Event");
        if (recEventBank.rows() != 1) {
            return StartTimeResult.invalid();
        }
        // The HIPO schema field is startTime (capital T).
        double value = recEventBank.getFloat("startTime", 0);
        return Double.isFinite(value)
                ? StartTimeResult.valid(value)
                : StartTimeResult.invalid();
    }

    private static boolean allFinite(double... values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private static boolean fitsFloat(double value) {
        return Double.isFinite(value)
                && value <= Float.MAX_VALUE
                && value >= -Float.MAX_VALUE;
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    private static final class StartTimeResult {
        private final boolean valid;
        private final double value;

        private StartTimeResult(boolean valid, double value) {
            this.valid = valid;
            this.value = value;
        }

        private static StartTimeResult valid(double value) {
            return new StartTimeResult(true, value);
        }

        private static StartTimeResult invalid() {
            return new StartTimeResult(false, Double.NaN);
        }
    }
}
