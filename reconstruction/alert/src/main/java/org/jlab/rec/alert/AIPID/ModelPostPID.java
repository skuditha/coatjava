package org.jlab.rec.alert.AIPID;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Logger;
import org.jlab.utils.CLASResources;

/**
 * DJL wrapper for the exported ALERT post-KF PID TorchScript model.
 *
 * <p>The model accepts raw engineered float32 features with shape [1, 18]. It
 * embeds standardization and softmax, and returns five probabilities in model
 * order [2212, 45, 46, 49, 47].</p>
 */
public final class ModelPostPID implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(ModelPostPID.class.getName());
    private static final String DEFAULT_RESOURCE_DIRECTORY =
            "etc/data/nnet/rg-l/model_PostPID/";
    private static final float PROBABILITY_SUM_TOLERANCE = 1.0e-3f;

    private final ZooModel<float[], float[]> model;
    private final Predictor<float[], float[]> predictor;

    /** Loads the production resource-directory model. */
    public ModelPostPID() {
        this(Paths.get(CLASResources.getResourcePath(DEFAULT_RESOURCE_DIRECTORY)));
    }

    /**
     * Loads a TorchScript model from an explicit file or artifact directory.
     *
     * @param modelPath path to model.pt or its containing directory
     */
    public ModelPostPID(Path modelPath) {
        if (modelPath == null) {
            throw new IllegalArgumentException("PostPID model path cannot be null");
        }

        Path normalizedPath = modelPath.toAbsolutePath().normalize();
        if (!Files.exists(normalizedPath)) {
            throw new IllegalArgumentException(
                    "PostPID model path does not exist: " + normalizedPath);
        }

        Translator<float[], float[]> translator = new Translator<>() {
            @Override
            public NDList processInput(TranslatorContext context, float[] features) {
                if (features == null || features.length != PostPIDContract.FEATURE_COUNT) {
                    throw new IllegalArgumentException(
                            "PostPID input must be float["
                                    + PostPIDContract.FEATURE_COUNT + "]");
                }
                NDManager manager = context.getNDManager();
                NDArray input = manager.create(
                        features,
                        new Shape(1, PostPIDContract.FEATURE_COUNT));
                return new NDList(input);
            }

            @Override
            public float[] processOutput(TranslatorContext context, NDList outputList) {
                if (outputList == null || outputList.isEmpty()) {
                    throw new IllegalStateException("PostPID model returned no output tensor");
                }
                float[] probabilities = outputList.get(0).toFloatArray();
                validateProbabilities(probabilities);
                return probabilities;
            }
        };

        System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
        System.setProperty("ai.djl.pytorch.num_threads", "1");
        System.setProperty("ai.djl.pytorch.graph_optimizer", "false");

        Criteria<float[], float[]> criteria = Criteria.builder()
                .setTypes(float[].class, float[].class)
                .optModelPath(normalizedPath)
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        try {
            model = criteria.loadModel();
            predictor = model.newPredictor();
            LOGGER.info(() -> "Loaded ALERT PostPID model from " + normalizedPath);
        } catch (IOException | ModelNotFoundException | MalformedModelException exception) {
            throw new IllegalStateException(
                    "Unable to load ALERT PostPID model from " + normalizedPath,
                    exception);
        }
    }

    /**
     * Runs inference for one resolved candidate.
     *
     * <p>No Java-side normalization or softmax is applied.</p>
     */
    public synchronized PostPIDResult prediction(PostPIDCandidate candidate)
            throws TranslateException {
        if (candidate == null) {
            throw new IllegalArgumentException("PostPID candidate cannot be null");
        }

        float[] probabilities = predictor.predict(candidate.getFeatures());
        validateProbabilities(probabilities);

        int bestIndex = 0;
        for (int index = 1; index < probabilities.length; index++) {
            if (probabilities[index] > probabilities[bestIndex]) {
                bestIndex = index;
            }
        }

        int pid = PostPIDContract.pidForClassIndex(bestIndex);

        // Model order is [2212, 45, 46, 49, 47].
        // Result fields are named in HIPO bank order [2212, 45, 46, 47, 49].
        return new PostPIDResult(
                candidate.trackid,
                candidate.clusterid,
                pid,
                probabilities[0],
                probabilities[1],
                probabilities[2],
                probabilities[4],
                probabilities[3]);
    }

    public ZooModel<float[], float[]> getModel() {
        return model;
    }

    @Override
    public void close() {
        predictor.close();
        model.close();
    }

    private static void validateProbabilities(float[] probabilities) {
        if (probabilities == null || probabilities.length != PostPIDContract.CLASS_COUNT) {
            throw new IllegalStateException(
                    "PostPID output must contain "
                            + PostPIDContract.CLASS_COUNT + " probabilities");
        }

        float sum = 0.0f;
        for (float probability : probabilities) {
            if (!Float.isFinite(probability) || probability < 0.0f || probability > 1.0f) {
                throw new IllegalStateException(
                        "PostPID model returned an invalid probability: " + probability);
            }
            sum += probability;
        }
        if (!Float.isFinite(sum) || Math.abs(sum - 1.0f) > PROBABILITY_SUM_TOLERANCE) {
            throw new IllegalStateException(
                    "PostPID probabilities do not sum to one: " + sum);
        }
    }
}
