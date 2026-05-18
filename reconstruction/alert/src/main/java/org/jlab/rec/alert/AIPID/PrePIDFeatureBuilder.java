package org.jlab.rec.alert.AIPID;

import java.util.HashSet;
import java.util.Set;
import org.jlab.io.base.DataBank;

/**
 * Builds the ALERT PrePID model input vector in the exact feature order used by
 * the improved alert_prepid training pipeline.
 *
 * <p>The deployed TorchScript model consumes 61 raw features. Standardization is
 * embedded inside the exported model, so this class must not standardize the
 * values before inference.</p>
 */
public final class PrePIDFeatureBuilder {

    public static final int INPUT_SIZE = 61;

    private static final double C_MM_PER_NS = 299.792458;
    private static final double ATOF_TIME_OFFSET_NS = 124.0;

    private static final double MASS_PROTON_GEV   = 0.9382720813;
    private static final double MASS_DEUTERON_GEV = 1.8756129426;
    private static final double MASS_TRITON_GEV   = 2.80892113298;
    private static final double MASS_HE3_GEV      = 2.80839160743;
    private static final double MASS_HE4_GEV      = 3.7273794066;

    private PrePIDFeatureBuilder() {
        // Utility class.
    }

    public static MatchResolution resolveTrackMatch(DataBank projections, DataBank hits, int trackid) {
        MatchResolution result = new MatchResolution();

        if (projections == null || hits == null) {
            return result;
        }

        for (int ip = 0; ip < projections.rows(); ip++) {
            if (projections.getInt("trackid", ip) != trackid) {
                continue;
            }

            final int hitId = projections.getInt("matched_atof_hit_id", ip);
            if (hitId < 0) {
                continue;
            }

            final int hitRow = findRowByInt(hits, "id", hitId);
            if (hitRow < 0) {
                result.invalidHitRefCount++;
                continue;
            }

            result.validMatchCount++;
            if (!result.hasValidMatch) {
                result.hasValidMatch = true;
                result.hitId = hitId;
                result.hitRow = hitRow;
            }
        }

        return result;
    }

    public static int findRowByInt(DataBank bank, String column, int value) {
        if (bank == null) {
            return -1;
        }
        for (int row = 0; row < bank.rows(); row++) {
            if (bank.getInt(column, row) == value) {
                return row;
            }
        }
        return -1;
    }

    /**
     * Build the 61-feature PrePID input row for one AHDC track.
     *
     * @param tracks AHDC::track bank
     * @param trackRow row in AHDC::track
     * @param hits ATOF::hits bank
     * @param match result of resolving ALERT::ai:projections for this track
     * @return raw 61-feature vector in model order
     */
    public static float[] buildFeatures(DataBank tracks, int trackRow, DataBank hits, MatchResolution match) {
        final float[] x = new float[INPUT_SIZE];

        final double tx = tracks.getFloat("x", trackRow);
        final double ty = tracks.getFloat("y", trackRow);
        final double tz = tracks.getFloat("z", trackRow);
        final double px = tracks.getFloat("px", trackRow);
        final double py = tracks.getFloat("py", trackRow);
        final double pz = tracks.getFloat("pz", trackRow);
        final int nHits = tracks.getInt("n_hits", trackRow);
        final int sumAdc = tracks.getInt("sum_adc", trackRow);
        final double dedx = tracks.getFloat("dEdx", trackRow);
        final double chi2 = tracks.getFloat("chi2", trackRow);

        final double pT = Math.sqrt(px * px + py * py);
        final double p = Math.sqrt(px * px + py * py + pz * pz);
        final double theta = Math.atan2(pT, pz);
        final double phi = Math.atan2(py, px);
        // Defaults used when there is no valid ATOF match.
        double atofTime = 0.0;
        double hx = 0.0;
        double hy = 0.0;
        double hz = 0.0;
        double hitEnergy = 0.0;
        double flightPathMm = 0.0;
        int sector = 0;
        int layer = 0;
        int component = 0;
        int clusterNHits = 0;
        int clusterUniqueLayers = 0;
        double clusterTotalEnergy = 0.0;
        double clusterMeanTime = 0.0;
        double clusterTimeSpread = 0.0;
        double clusterEnergyWeightedTime = 0.0;
        double clusterMeanX = 0.0;
        double clusterMeanY = 0.0;
        double clusterMeanZ = 0.0;
        double clusterEnergyWeightedX = 0.0;
        double clusterEnergyWeightedY = 0.0;
        double clusterEnergyWeightedZ = 0.0;
        int hasAtofMatch = 0;

        if (match != null && match.hasValidMatch && hits != null && match.hitRow >= 0) {
            hasAtofMatch = 1;

            sector = hits.getInt("sector", match.hitRow);
            layer = hits.getInt("layer", match.hitRow);
            component = hits.getInt("component", match.hitRow);
            final int clusterId = hits.getInt("clusterid", match.hitRow);
            final double rawTime = hits.getFloat("time", match.hitRow);

            atofTime = rawTime - ATOF_TIME_OFFSET_NS;
            hx = hits.getFloat("x", match.hitRow);
            hy = hits.getFloat("y", match.hitRow);
            hz = hits.getFloat("z", match.hitRow);
            hitEnergy = hits.getFloat("energy", match.hitRow);

            final double dx = hx - tx;
            final double dy = hy - ty;
            final double dz = hz - tz;
            flightPathMm = Math.sqrt(dx * dx + dy * dy + dz * dz);

            ClusterSummary cluster = summarizeCluster(hits, clusterId);
            clusterNHits = cluster.nHits;
            clusterUniqueLayers = cluster.uniqueLayers;
            clusterTotalEnergy = cluster.totalEnergy;
            clusterMeanTime = cluster.meanTime;
            clusterTimeSpread = cluster.timeSpread;
            clusterEnergyWeightedTime = cluster.energyWeightedTime;
            clusterMeanX = cluster.meanX;
            clusterMeanY = cluster.meanY;
            clusterMeanZ = cluster.meanZ;
            clusterEnergyWeightedX = cluster.energyWeightedX;
            clusterEnergyWeightedY = cluster.energyWeightedY;
            clusterEnergyWeightedZ = cluster.energyWeightedZ;
        }

        final double beta = (Double.isFinite(flightPathMm) && Double.isFinite(atofTime) && Math.abs(atofTime) > 1.0e-12)
                ? flightPathMm / (C_MM_PER_NS * atofTime)
                : 0.0;
        final boolean betaValid = Double.isFinite(beta) && Math.abs(beta) > 1.0e-12;
        final double invBeta = betaValid ? 1.0 / beta : 0.0;
        final double invBeta2 = betaValid ? invBeta * invBeta : 0.0;

        final double rigidityGev = p / 1000.0;
        final double mass2Term = betaValid ? invBeta2 - 1.0 : 0.0;
        final double mass2Q1 = rigidityGev * rigidityGev * mass2Term;
        final double mass2Q2 = (2.0 * rigidityGev) * (2.0 * rigidityGev) * mass2Term;

        final float bProton = expectedBeta(rigidityGev, 1, MASS_PROTON_GEV);
        final float bDeuteron = expectedBeta(rigidityGev, 1, MASS_DEUTERON_GEV);
        final float bTriton = expectedBeta(rigidityGev, 1, MASS_TRITON_GEV);
        final float bHe3 = expectedBeta(rigidityGev, 2, MASS_HE3_GEV);
        final float bHe4 = expectedBeta(rigidityGev, 2, MASS_HE4_GEV);

        final double logDedx = safeLog1p(dedx);

        x[0]  = finiteFloat(tx);                                     // track_x
        x[1]  = finiteFloat(ty);                                     // track_y
        x[2]  = finiteFloat(tz);                                     // track_z
        x[3]  = finiteFloat(px);                                     // track_px
        x[4]  = finiteFloat(py);                                     // track_py
        x[5]  = finiteFloat(pz);                                     // track_pz
        x[6]  = nHits;                                               // n_hits
        x[7]  = sumAdc;                                              // sum_adc
        x[8]  = finiteFloat(flightPathMm);                           // path
        x[9]  = finiteFloat(dedx);                                   // dEdx
        x[10] = finiteFloat(chi2);                                   // chi2
        x[11] = finiteFloat(atofTime);                               // atof_time
        x[12] = finiteFloat(hx);                                     // hit_x
        x[13] = finiteFloat(hy);                                     // hit_y
        x[14] = finiteFloat(hz);                                     // hit_z
        x[15] = finiteFloat(hitEnergy);                              // hit_energy
        x[16] = finiteFloat(p);                                      // p_over_q
        x[17] = finiteFloat(safeLog1p(p));                           // log1p_p_over_q
        x[18] = finiteFloat(pT);                                     // pT
        x[19] = finiteFloat(theta);                                  // theta
        x[20] = finiteFloat(phi);                                    // phi
        x[21] = finiteFloat(safeRatio(sumAdc, nHits));               // sum_adc_per_hit
        x[22] = finiteFloat(safeRatio(chi2, nHits));                 // chi2_per_hit
        x[23] = finiteFloat(safeLog1p(sumAdc));                      // log1p_sum_adc
        x[24] = finiteFloat(logDedx);                                // log1p_dEdx
        x[25] = finiteFloat(safeLog1p(chi2));                        // log1p_chi2
        x[26] = finiteFloat(betaValid ? beta : 0.0);                 // beta
        x[27] = finiteFloat(invBeta);                                // inv_beta
        x[28] = finiteFloat(invBeta2);                               // inv_beta2
        x[29] = finiteFloat(mass2Q1);                                // mass2_q1
        x[30] = signedLog1pAbs(mass2Q1);                             // signed_log1p_abs_mass2_q1
        x[31] = finiteFloat(mass2Q2);                                // mass2_q2
        x[32] = signedLog1pAbs(mass2Q2);                             // signed_log1p_abs_mass2_q2
        x[33] = bProton > 0.0f ? finiteFloat(invBeta - 1.0 / bProton) : 0.0f;
        x[34] = bDeuteron > 0.0f ? finiteFloat(invBeta - 1.0 / bDeuteron) : 0.0f;
        x[35] = bTriton > 0.0f ? finiteFloat(invBeta - 1.0 / bTriton) : 0.0f;
        x[36] = bHe3 > 0.0f ? finiteFloat(invBeta - 1.0 / bHe3) : 0.0f;
        x[37] = bHe4 > 0.0f ? finiteFloat(invBeta - 1.0 / bHe4) : 0.0f;
        x[38] = finiteFloat(dedx * beta * beta);                     // dedx_times_beta2
        x[39] = betaValid ? finiteFloat(dedx / (beta * beta)) : 0.0f; // dedx_over_beta2
        x[40] = finiteFloat(safeLog1p(hitEnergy));                   // log1p_hit_energy
        x[41] = finiteFloat(hitEnergy * beta * beta);                // hit_energy_times_beta2
        x[42] = finiteFloat(p * beta);                               // p_times_beta
        x[43] = betaValid ? finiteFloat(p / beta) : 0.0f;            // p_over_beta
        x[44] = finiteFloat(p * logDedx);                            // p_over_q_times_log1p_dEdx
        x[45] = finiteFloat(clusterTotalEnergy);                     // cluster_total_energy
        x[46] = finiteFloat(clusterMeanTime);                        // cluster_mean_time
        x[47] = finiteFloat(clusterTimeSpread);                      // cluster_time_spread
        x[48] = finiteFloat(clusterEnergyWeightedTime);              // cluster_energy_weighted_time
        x[49] = finiteFloat(clusterMeanX);                           // cluster_mean_x
        x[50] = finiteFloat(clusterMeanY);                           // cluster_mean_y
        x[51] = finiteFloat(clusterMeanZ);                           // cluster_mean_z
        x[52] = finiteFloat(clusterEnergyWeightedX);                 // cluster_energy_weighted_x
        x[53] = finiteFloat(clusterEnergyWeightedY);                 // cluster_energy_weighted_y
        x[54] = finiteFloat(clusterEnergyWeightedZ);                 // cluster_energy_weighted_z
        x[55] = sector;                                              // atof_sector
        x[56] = layer;                                               // atof_layer
        x[57] = component;                                           // atof_component
        x[58] = clusterNHits;                                        // cluster_n_hits
        x[59] = clusterUniqueLayers;                                 // cluster_unique_layers
        x[60] = hasAtofMatch;                                        // has_atof_match

        return x;
    }

    private static ClusterSummary summarizeCluster(DataBank hits, int clusterId) {
        ClusterSummary summary = new ClusterSummary();
        if (hits == null || clusterId < 0) {
            return summary;
        }

        double sumTimeRaw = 0.0;
        double minTimeRaw = Double.POSITIVE_INFINITY;
        double maxTimeRaw = Double.NEGATIVE_INFINITY;
        double sumX = 0.0;
        double sumY = 0.0;
        double sumZ = 0.0;
        double ewTimeRaw = 0.0;
        double ewX = 0.0;
        double ewY = 0.0;
        double ewZ = 0.0;
        Set<Integer> uniqueLayers = new HashSet<>();

        for (int row = 0; row < hits.rows(); row++) {
            if (hits.getInt("clusterid", row) != clusterId) {
                continue;
            }

            final double energy = hits.getFloat("energy", row);
            final double timeRaw = hits.getFloat("time", row);
            final double x = hits.getFloat("x", row);
            final double y = hits.getFloat("y", row);
            final double z = hits.getFloat("z", row);

            summary.nHits++;
            summary.totalEnergy += energy;
            sumTimeRaw += timeRaw;
            minTimeRaw = Math.min(minTimeRaw, timeRaw);
            maxTimeRaw = Math.max(maxTimeRaw, timeRaw);
            sumX += x;
            sumY += y;
            sumZ += z;
            ewTimeRaw += energy * timeRaw;
            ewX += energy * x;
            ewY += energy * y;
            ewZ += energy * z;
            uniqueLayers.add(hits.getInt("layer", row));
        }

        if (summary.nHits <= 0) {
            return summary;
        }

        final double meanTimeRaw = sumTimeRaw / summary.nHits;
        final double ewTimeRawNorm = summary.totalEnergy > 0.0 ? ewTimeRaw / summary.totalEnergy : meanTimeRaw;

        summary.uniqueLayers = uniqueLayers.size();
        summary.meanTime = meanTimeRaw - ATOF_TIME_OFFSET_NS;
        summary.timeSpread = maxTimeRaw - minTimeRaw;
        summary.energyWeightedTime = ewTimeRawNorm - ATOF_TIME_OFFSET_NS;
        summary.meanX = sumX / summary.nHits;
        summary.meanY = sumY / summary.nHits;
        summary.meanZ = sumZ / summary.nHits;
        summary.energyWeightedX = summary.totalEnergy > 0.0 ? ewX / summary.totalEnergy : summary.meanX;
        summary.energyWeightedY = summary.totalEnergy > 0.0 ? ewY / summary.totalEnergy : summary.meanY;
        summary.energyWeightedZ = summary.totalEnergy > 0.0 ? ewZ / summary.totalEnergy : summary.meanZ;

        return summary;
    }

    private static float expectedBeta(double rigidityGev, int absCharge, double massGev) {
        final double pGev = Math.abs(absCharge) * rigidityGev;
        if (!Double.isFinite(pGev) || pGev <= 0.0 || massGev <= 0.0) {
            return 0.0f;
        }
        return finiteFloat(pGev / Math.sqrt(pGev * pGev + massGev * massGev));
    }

    private static float safeLog1p(double value) {
        if (!Double.isFinite(value) || value <= -1.0) {
            return 0.0f;
        }
        return finiteFloat(Math.log1p(value));
    }

    private static float signedLog1pAbs(double value) {
        if (!Double.isFinite(value)) {
            return 0.0f;
        }
        final double sign = value < 0.0 ? -1.0 : 1.0;
        return finiteFloat(sign * Math.log1p(Math.abs(value)));
    }

    private static float safeRatio(double numerator, double denominator) {
        if (!Double.isFinite(numerator) || !Double.isFinite(denominator) || Math.abs(denominator) < 1.0e-12) {
            return 0.0f;
        }
        return finiteFloat(numerator / denominator);
    }

    private static float finiteFloat(double value) {
        if (!Double.isFinite(value)) {
            return 0.0f;
        }
        return (float) value;
    }

    public static final class MatchResolution {
        public boolean hasValidMatch = false;
        public int hitId = -1;
        public int hitRow = -1;
        public int validMatchCount = 0;
        public int invalidHitRefCount = 0;
    }

    private static final class ClusterSummary {
        int nHits = 0;
        int uniqueLayers = 0;
        double totalEnergy = 0.0;
        double meanTime = 0.0;
        double timeSpread = 0.0;
        double energyWeightedTime = 0.0;
        double meanX = 0.0;
        double meanY = 0.0;
        double meanZ = 0.0;
        double energyWeightedX = 0.0;
        double energyWeightedY = 0.0;
        double energyWeightedZ = 0.0;
    }
}
