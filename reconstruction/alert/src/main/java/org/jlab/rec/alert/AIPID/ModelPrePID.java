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
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.jlab.utils.CLASResources;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.logging.Logger;

public class ModelPrePID {

    static final Logger LOGGER = Logger.getLogger(ModelPrePID.class.getName());

    // Update to match the improved training class order:
    // proton, deuteron, triton, helium3, helium4.
    private static final int[] CLASS_IDS = new int[]{2212, 45, 46, 49, 47};

    private final ZooModel<float[], float[]> model;

    public ModelPrePID() {

        Translator<float[], float[]> translator = new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext ctx, float[] floats) {
                NDManager manager = ctx.getNDManager();

                // The improved TorchScript PrePID model expects one raw 61-feature row.
                // Standardization is embedded in the exported model.
                NDArray x = manager.create(floats, new Shape(1, PrePIDFeatureBuilder.INPUT_SIZE));
                return new NDList(x);
            }

            @Override
            public Batchifier getBatchifier() {
                return null;
            }

            @Override
            public float[] processOutput(TranslatorContext ctx, NDList ndList) {
                NDArray logits = ndList.get(0);      // (1,5), model class order
                NDArray probs = logits.softmax(1);   // convert logits to probabilities

                float[] p = probs.toFloatArray();    // model order: p, d, t, he3, he4

                int bestIdx = 0;
                float best = p[0];
                for (int k = 1; k < p.length; k++) {
                    if (p[k] > best) {
                        best = p[k];
                        bestIdx = k;
                    }
                }
                int prepid = CLASS_IDS[bestIdx];

                // Return the bank order expected by ALERT::ai:prepid:
                // prepid, p2212, p45, p46, p47, p49.
                return new float[]{
                    (float) prepid,
                    p[0], // p2212: proton
                    p[1], // p45: deuteron
                    p[2], // p46: triton
                    p[4], // p47: helium4
                    p[3]  // p49: helium3
                };
            }
        };

        System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
        System.setProperty("ai.djl.pytorch.num_threads", "1");
        System.setProperty("ai.djl.pytorch.graph_optimizer", "false");

        String path = CLASResources.getResourcePath("etc/data/nnet/rg-l/model_PrePID/");

        Criteria<float[], float[]> criteria = Criteria.builder()
                .setTypes(float[].class, float[].class)
                .optModelPath(Paths.get(path))
                .optModelName("model_PrePID")
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        try {
            model = criteria.loadModel();
        } catch (IOException | ModelNotFoundException | MalformedModelException e) {
            throw new RuntimeException(e);
        }
    }

    public ZooModel<float[], float[]> getModel() {
        return model;
    }

    /**
     * Returns float[]{prepid, p2212, p45, p46, p47, p49}.
     *
     * @param features61 raw 61-feature vector in the improved PrePID order
     * @return PID prediction and probabilities in ALERT::ai:prepid bank order
     * @throws ai.djl.translate.TranslateException if DJL inference fails
     */
    public float[] prediction(float[] features61) throws TranslateException {
        if (features61 == null || features61.length != PrePIDFeatureBuilder.INPUT_SIZE) {
            LOGGER.warning("PrePID input must be float[" + PrePIDFeatureBuilder.INPUT_SIZE + "]");
            return null;
        }
        try (Predictor<float[], float[]> predictor = model.newPredictor()) {
            return predictor.predict(features61);
        }
    }
}
