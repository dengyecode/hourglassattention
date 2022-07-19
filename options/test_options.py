from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='/data/dy/code/inpainting/utrans/z/new/6', help='saves results here')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')

        self.isTrain = False

        return parser
