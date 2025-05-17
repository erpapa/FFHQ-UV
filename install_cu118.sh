# install torch 1.7.1
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 --index-url https://download.pytorch.org/whl/cu118

# pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
# install dependencies
pip install -v -r requirements.txt

cd ../
git clone https://github.com/facebookresearch/iopath
git clone https://github.com/facebookresearch/fvcore
git clone -b v0.7.2 https://github.com/facebookresearch/pytorch3d
git clone https://github.com/NVlabs/nvdiffrast

conda install -c bottler nvidiacub
pip install -v ./ iopath
pip install -v ./ fvcore
pip install -v ./ pytorch3d
pip install -v ./ nvdiffrast

git clone https://github.com/adriansahlman/stylegan2_pytorch.git
cd stylegan2_pytorch
python run_convert_from_tf.py --input="../FFHQ/checkpoints/stylegan_model/stylegan2-ffhq-config-f.pkl" --output G.pth D.pth Gs.pth
mv G.pth ../FFHQ/checkpoints/stylegan_model/stylegan2-ffhq-config-f-G.pth
mv D.pth ../FFHQ/checkpoints/stylegan_model/stylegan2-ffhq-config-f-D.pth
mv Gs.pth ../FFHQ/checkpoints/stylegan_model/stylegan2-ffhq-config-f-Gs.pth
