import argparse
import cv2
import numpy as np
import torch
from marlin_pytorch import Marlin
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from model.classifier import Classifier

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


# def reshape_transform(tensor, height=14, width=14):
#     result = tensor[:, 1:, :].reshape(tensor.size(0),
#                                       height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result
def reshape_transform(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0), 8 ,height, width, tensor.size(2))
    result = result.permute(0, 1 , 4, 2 , 3)
    return result

if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    model = Marlin.from_file('marlin_vit_small_ytf', '/home/duke/Workspace/MARLIN/ckpt/marlin_vit_small/last-v1.ckpt').encoder
    # model = Marlin.from_online('marlin_vit_base_ytf').encoder
    # model = Classifier(3, 'marlin_vit_small_ytf', True)
    
    # # target_layers = [model.blocks[-1]]
    # # Step 2: Load the checkpoint (replace 'path_to_checkpoint.ckpt' with the actual path)
    # checkpoint = torch.load('/home/duke/Workspace/MARLIN/ckpt/BiovidBBBBBBBBBBB_full_video-epoch=8-val_acc=0.5124-val_auc=0.5124.ckpt')
    print(model)
    # # Step 3: Load the model state dictionary
    # model.load_state_dict(checkpoint['model_state_dict'])
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    print(input_tensor.shape)
    input_tensor = input_tensor.repeat(16, 1, 1, 1).permute(1,0,2,3).unsqueeze(0)
    
    print(input_tensor.shape)
    # input_tensor = torch.rand(4,3,16,224,224)
    # print('ffffffffffffffffffffffffffff')
    print('dfdfdfdf', model)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[-2].norm2]
    # target_layers = [model.model.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = cv2.resize(rgb_img, (224, 224))
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
    #                                 std=[0.5, 0.5, 0.5])
    # print('xxxxxxxxxxxxxxxxxx', input_tensor.shape)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :][0]
    print('grayscale_cam', grayscale_cam.shape)
    # grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

    # Step 2: Apply gamma correction to enhance warm areas
    gamma = 0.75  # Higher values of gamma make warm colors more pronounced
    grayscale_cam = grayscale_cam ** gamma

    # # Step 3: Optionally scale up the bright areas further by applying another factor
    scaling_factor = 1.7
    grayscale_cam = grayscale_cam * scaling_factor
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)

    