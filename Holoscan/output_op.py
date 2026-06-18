# =========================================================
# output_op.py
# OutputOp -- records results, measures latency, prints
# classification with confidence for each frame.
# v1.7: predictions are always benign or malignant only
# (normal probability is absorbed into benign by InferenceOp)
# =========================================================

import holoscan
import numpy as np
import time


class OutputOp(holoscan.core.Operator):

    def setup(self, spec):
        spec.input('scores')      # [3] float32 raw probabilities (benign/malignant/normal)
        spec.input('prediction')  # predicted class string: 'benign' or 'malignant'
        spec.input('label')       # ground truth: 0=malignant, 1=benign

    def start(self):
        self.frame_count = 0
        self.correct     = 0
        self.results     = []
        self.start_time  = time.time()
        self.frame_times = []
        self.last_time   = time.time()

        # Per-class counters for summary
        self.tp_mal = 0   # true positive malignant
        self.fn_mal = 0   # false negative malignant (missed cancer)
        self.tp_ben = 0   # true positive benign
        self.fn_ben = 0   # false negative benign

        print('\n' + '=' * 64)
        print('  Medical Imaging Pipeline v1.7 -- Results')
        print('=' * 64)
        print(f"{'Frame':<6} {'True':<12} {'Predicted':<12} {'Confidence':>10}  {'OK?'}")
        print('-' * 64)

    def compute(self, op_input, op_output, context):
        scores     = op_input.receive('scores')
        prediction = op_input.receive('prediction')
        label      = op_input.receive('label')

        now = time.time()
        frame_latency = (now - self.last_time) * 1000
        self.frame_times.append(frame_latency)
        self.last_time = now

        true_class = 'benign' if label == 1 else 'malignant'
        is_correct = (prediction == true_class)

        if is_correct:
            self.correct += 1

        # Per-class tracking
        if true_class == 'malignant':
            if is_correct: self.tp_mal += 1
            else:          self.fn_mal += 1
        else:
            if is_correct: self.tp_ben += 1
            else:          self.fn_ben += 1

        self.frame_count += 1

        # Confidence: use merged binary probability for the predicted class
        p_benign_merged = float(scores[0]) + float(scores[2])
        p_malignant     = float(scores[1])
        confidence      = (p_malignant if prediction == 'malignant' else p_benign_merged) * 100

        tick = 'OK' if is_correct else '--'
        print(f'{self.frame_count:<6} {true_class:<12} {prediction:<12} '
              f'{confidence:>9.1f}%  {tick}')

        self.results.append({
            'frame':      self.frame_count,
            'true':       true_class,
            'predicted':  prediction,
            'confidence': confidence,
            'correct':    is_correct,
            'latency_ms': frame_latency,
            'scores':     scores.tolist(),
        })

    def stop(self):
        total_time  = time.time() - self.start_time
        accuracy    = (self.correct / self.frame_count * 100) if self.frame_count > 0 else 0
        avg_latency = np.mean(self.frame_times[1:]) if len(self.frame_times) > 1 else 0
        fps         = self.frame_count / total_time if total_time > 0 else 0

        mal_recall  = self.tp_mal / (self.tp_mal + self.fn_mal + 1e-9) * 100
        ben_recall  = self.tp_ben / (self.tp_ben + self.fn_ben + 1e-9) * 100

        print('=' * 64)
        print('  Pipeline Summary')
        print('=' * 64)
        print(f'  Frames processed    : {self.frame_count}')
        print(f'  Overall accuracy    : {accuracy:.1f}%  ({self.correct}/{self.frame_count})')
        print(f'  Malignant recall    : {mal_recall:.1f}%  '
              f'({self.tp_mal}/{self.tp_mal+self.fn_mal}  caught)')
        print(f'  Benign recall       : {ben_recall:.1f}%  '
              f'({self.tp_ben}/{self.tp_ben+self.fn_ben}  caught)')
        print(f'  Missed cancers      : {self.fn_mal}')
        print(f'  Avg frame latency   : {avg_latency:.1f} ms')
        print(f'  Throughput          : {fps:.1f} fps')
        print(f'  Total time          : {total_time:.2f} s')
        print('=' * 64)

        np.save('/home/rohit/project/pipeline_results.npy', self.results)
        print('Results saved to ~/project/pipeline_results.npy')