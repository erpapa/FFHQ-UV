import os
import time
import torch
import pickle
import argparse
import numpy as np
import PIL.Image as Image

import stylegan2
from stylegan2 import utils
# import dnnlib as dnnlib
# import dnnlib.tflib as tflib


# def load_networks(path):
#     stream = open(path, 'rb')
#     tflib.init_tf()
#     with stream:
#         G, D, Gs = pickle.load(stream, encoding='latin1')
#     return G, D, Gs


class StyleGAN2_Model:

    def __init__(self, network_pkl):

        print('Loading networks from "%s"...' % network_pkl)
        self.Gs = stylegan2.models.load(network_pkl.replace('.pkl', '-Gs.pth'))
        self.Gs.eval()
        self.Gs.to('cuda')
        
        # noise_buffer
        named_buffers = self.Gs.G_synthesis.named_buffers()
        self.noise_bufs = {name: buf for (name, buf) in named_buffers if 'noise_const' in name}
        # noise_buffer keep
        self.noise_bufs_keep = {name: buf.detach().clone() for (name, buf) in self.noise_bufs.items()}
        # Randomize noise buffers.
        seed = 0
        latent_size, label_size = self.Gs.latent_size, self.Gs.label_size
        noise_reference = self.Gs.static_noise()
        _latents, _labels, noise_tensors = self.generate_batch(latent_size, label_size, noise_reference, [seed])
        self.Gs.static_noise(noise_tensors=noise_tensors)

        # _G, _D, Gs = load_networks(network_pkl)
        # self.Gs = Gs
        # self.Gs_syn_kwargs = dnnlib.EasyDict()
        # self.Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # self.Gs_syn_kwargs.randomize_noise = False
        # self.Gs_syn_kwargs.minibatch_size = 4
        # self.noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        # rnd = np.random.RandomState(0)
        # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars})

    def generate_batch(self, latent_size, label_size, noise_reference, seeds):
        latents = []
        labels = []
        noise_tensors = [[] for _ in noise_reference]
        for seed in seeds:
            rnd = np.random.RandomState(seed)
            latents.append(torch.from_numpy(rnd.randn(latent_size)))
            for i, ref in enumerate(noise_reference):
                noise_tensors[i].append(torch.from_numpy(rnd.randn(*ref.size()[1:])))
            if label_size:
                labels.append(torch.tensor([rnd.randint(0, label_size)]))
        latents = torch.stack(latents, dim=0).to(device='cuda', dtype=torch.float32)
        if label_size:
            labels = torch.cat(labels, dim=0).to(device='cuda', dtype=torch.int64)
        else:
            labels = None
        noise_tensors = [
            torch.stack(noise, dim=0).to(device='cuda', dtype=torch.float32)
            for noise in noise_tensors
        ]
        return latents, labels, noise_tensors
    
    def generate_im_from_random_seed(self, seed=22, truncation_psi=0.5):
        Gs = self.Gs
        # Randomize noise buffers.
        latent_size, label_size = Gs.latent_size, Gs.label_size
        noise_reference = Gs.static_noise()
        latents, labels, noise_tensors = self.generate_batch(latent_size, label_size, noise_reference, [seed])
        if truncation_psi is not None:
            Gs.set_truncation(truncation_psi=truncation_psi)
        if noise_tensors is not None:
            Gs.static_noise(noise_tensors=noise_tensors)
        with torch.no_grad():
            images = Gs(latents, labels=labels)
        # 后处理：将图像从[-1,1]范围转换到[0,255]并转为NumPy
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images
    
        # Gs = self.Gs
        # seeds = [seed]
        # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

        # Gs_kwargs = dnnlib.EasyDict()
        # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # Gs_kwargs.randomize_noise = False
        # if truncation_psi is not None:
        #     Gs_kwargs.truncation_psi = truncation_psi

        # for seed_idx, seed in enumerate(seeds):
        #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        #     rnd = np.random.RandomState(seed)
        #     z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        #     tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        #     images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        # return images

    def generate_im_from_z_space(self, z, truncation_psi=0.5):
        Gs = self.Gs
        if truncation_psi is not None:
            Gs.set_truncation(truncation_psi=truncation_psi)
        z_tensor = torch.from_numpy(z).to(torch.float32).to('cuda')
        # 检查维度：如果缺少batch维度，则添加
        if z_tensor.ndim == 2:  # 形状为 [num_layers, w_dim]
            z_tensor = z_tensor.unsqueeze(0)  # 添加batch维度 -> [1, num_layers, w_dim]
        # 生成图像
        with torch.no_grad():
            images = Gs(z_tensor)
        # 后处理：将图像从[-1,1]范围转换到[0,255]并转为NumPy
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images
    
        # Gs_kwargs = dnnlib.EasyDict()
        # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # Gs_kwargs.randomize_noise = False
        # if truncation_psi is not None:
        #     Gs_kwargs.truncation_psi = truncation_psi  # [height, width]

        # images = self.Gs.run(z, None, **Gs_kwargs)
        # return images

    def generate_im_from_w_space(self, w):
        w_tensor = torch.from_numpy(w).to(torch.float32).to('cuda')
        # 检查维度：如果缺少batch维度，则添加
        if w_tensor.ndim == 2:  # 形状为 [num_layers, w_dim]
            w_tensor = w_tensor.unsqueeze(0)  # 添加batch维度 -> [1, num_layers, w_dim]
        # 生成图像
        with torch.no_grad():
            images = self.Gs.G_synthesis(w_tensor)
        # 后处理：将图像从[-1,1]范围转换到[0,255]并转为NumPy
        images = (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images
    
        # images = self.Gs.components.synthesis.run(w, **self.Gs_syn_kwargs)
        # return images


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="gen_face_from_latent")
    parser.add_argument(
        "--latent_dir",
        type=str,
        required=True,
        help="The directory of the download multi-view face latent codes.",
    )
    parser.add_argument(
        "--save_face_dir",
        type=str,
        required=True,
        help="The directory for saving generated multi-view face images.",
    )
    parser.add_argument(
        "--stylegan_network_pkl",
        type=str,
        default='./stylegan2-ffhq-config-f.pkl',
        help="The path of the offical pretrained StyleGAN2 network pkl file.",
    )
    args = parser.parse_args()

    # ----------------------- Define Inference Parameters -----------------------
    latent_dir = args.latent_dir
    save_face_dir = args.save_face_dir
    stylegan_network_pkl = args.stylegan_network_pkl

    os.makedirs(save_face_dir, exist_ok=True)

    # ----------------------- Load StyleGAN Model -----------------------
    stylegan_model = StyleGAN2_Model(stylegan_network_pkl)

    # ----------------------- Perform Editing -----------------------
    fnames = sorted(os.listdir(latent_dir))
    for fn in fnames:
        tic = time.time()

        face_latent = torch.load(os.path.join(latent_dir, fn), map_location='cpu')

        face_image = stylegan_model.generate_im_from_w_space(face_latent.detach().cpu().numpy())[0]

        Image.fromarray(face_image, 'RGB').save(os.path.join(save_face_dir, f'{fn.split(".")[0]}.png'))

        toc = time.time()
        print('Generate {} done, took {:.4f} seconds.'.format(fn, toc - tic))


    print(f'Generate all done!')
