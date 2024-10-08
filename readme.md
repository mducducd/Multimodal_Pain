## mts env fix
echo "# Multimodel_Pain" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/mducducd/Multimodel_Pain.git
git push -u origin main
### TypeError: forward() got an unexpected keyword argument 'is_causal' (torch 2.x)
site-packpage -> torch(>2.) -> TransformerEncoder -> forward -> for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers) ##remove is_casual=is_casual


## Acknowledgements
Much of the code in this repo is taken from [MARLIN]([https://github.com/Mael-zys/T2M-GPT](https://github.com/ControlNet/MARLIN)).
