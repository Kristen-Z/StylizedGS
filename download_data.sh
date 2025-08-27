mkdir datasets
cd datasets

# download LLFF datasets
gdown 11PhkBXZZNYTD2emdG1awALlhCnkq7aN-
unzip nerf_llff_data.zip && mv nerf_llff_data llff && rm nerf_llff_data.zip

# download Tanks and Temples datasets


# download MipNeRF-360 datasets
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip && mv 360_v2 mipnerf360

# download style images
wget https://3romcg.bn.files.1drv.com/y4m_wXlfBvwva8k9WU4o0ZeFOACIdLr3dN1zxPtGKNmouKJp9jZW4P5K1dDY2T14xl4lAywbFquMxowH4QrgIC5sNnvOMzkviXvMeQqefFxyC05Xaeov3jNW4OfhaFE_-8rtWzyMTuGxuF5uM1XyE_BPzEXg_M7QKE1zn-fqniAOYYnBPj7GeEqQNDf744Wq2C8PJF7ibbc9S6fKWUKoUPtDQ -O style120_data.zip
unzip style120_data.zip && mv style120_data styles && rm style120_data.zip