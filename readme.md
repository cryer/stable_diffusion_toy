# stable diffusion toy

Implement a toy stable diffusion from scratch. Please focus on  code rather than result,cause it's only trained with tiny u-net and 1000 casual images in only one hour.

## main work

- simple u-net with time TimestepEmbedding , CrossAttention and space SelfAttention

- diffusion from scratch

- DDIMSampler  from scratch

- use pretrained vae and clip,cause they don't update in training period 

## mention

my dataset is casually  spidered online .if you want to train by your own,just use `LAION-5B` open dataset.Just choose 1000 or more if you want from it and rename as `1.jpg,2.jpg` to like `1000.jpg`,and paste in `dataset/images` directory.Also cations in `cations.txt` line by line correspondly.

## result

<table>
    <tr>
        <td><img src="https://github.com/cryer/stable_diffusion_toy/raw/master/doc/1.png" alt="Image 1" width="300"></td>
        <td><img src="https://github.com/cryer/stable_diffusion_toy/raw/master/doc/2.png" alt="Image 2" width="300"></td>
        <td><img src="https://github.com/cryer/stable_diffusion_toy/raw/master/doc/3.png" alt="Image 3" width="300"></td>
    </tr>
</table>

## other

If you want to learn more about stable diffusion and this toy repo, you can check my blog [chinese only]([实现一个简化版的stable diffusion](https://cryer.github.io/2022/10/sd/)).


