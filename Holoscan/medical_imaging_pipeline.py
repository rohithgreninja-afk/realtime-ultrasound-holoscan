# =========================================================
# medical_imaging_pipeline.py
# Full Holoscan pipeline -- v1.7
# Model: trainedMobileNetV2_mega.onnx
# Beamforming: A-line hilbert + power law gamma=0.3
# Normal handling: p_normal absorbed into p_benign (InferenceOp)
# Predictions: binary only (benign / malignant)
# Phase 6 benchmark: 34.0 fps mean, 29.37 ms avg latency, 73% accuracy
# =========================================================

import holoscan
from holoscan.conditions import CountCondition

from data_source_op  import DataSourceOp
from beamforming_op  import BeamformingOp
from enhancement_op  import EnhancementOp
from inference_op    import InferenceOp
from output_op       import OutputOp


class MedicalImagingApp(holoscan.core.Application):

    def __init__(self, num_frames=100):
        super().__init__()
        self.num_frames = num_frames

    def compose(self):
        source   = DataSourceOp(self,
                       CountCondition(self, count=self.num_frames),
                       name='source')
        beamform = BeamformingOp(self,  name='beamform')
        enhance  = EnhancementOp(self,  name='enhance')
        infer    = InferenceOp(self,    name='infer')
        output   = OutputOp(self,       name='output')

        self.add_flow(source,   beamform, {('rf_frame', 'rf_frame'),
                                           ('label',    'label')})
        self.add_flow(beamform, enhance,  {('bmode',    'bmode'),
                                           ('label',    'label')})
        self.add_flow(enhance,  infer,    {('enhanced', 'enhanced'),
                                           ('label',    'label')})
        self.add_flow(infer,    output,   {('scores',     'scores'),
                                           ('prediction', 'prediction'),
                                           ('label',      'label')})


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f'Starting Medical Imaging Pipeline v1.7 -- {n} frames')
    print(f'Model  : trainedMobileNetV2_mega.onnx')
    print(f'Beamform: A-line Hilbert + power law gamma=0.3')
    print(f'Output : binary (benign / malignant), normal absorbed into benign\n')
    app = MedicalImagingApp(num_frames=n)
    app.run()