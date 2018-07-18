#Configuration 추가. default값 바꾸는 코드
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', default='C:\\Users\JINI\workspace\DeepLearning\PROGRAPHY DATA', help='path to dataset')
# hyper-parameters
"""
epochs = 10   // epochs는 서버에서 돌려보고 조절.
learning_rate = 0.0002
"""
parser.add_argument('--log_interval', type=int, default=100, help='save valid gif and image')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--checkpoint_step', type=int, default=10, help='save checkpoint')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.999')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay for adam.default=0.00001')


# misc
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default=None, help='folder to output images and model checkpoints')
#pre-trained model path (https://github.com/kenshohara/3D-ResNets-PyTorch/releases)
parser.add_argument('--pretrained_path', default="C:/Users/JINI/workspace/DeepLearning/c3d.pickle", help="path to pre-trained ResNet model")


def get_config():
    return parser.parse_args()
