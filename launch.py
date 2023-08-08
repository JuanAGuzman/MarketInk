import gradio as gr
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms as tfms
import accelerate
import os
import replicate
from datetime import datetime
os.getcwd()
print(os.getcwd())
# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

#huggingface-cli login

# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);
model = replicate.models.get("stability-ai/stable-diffusion")

def eval_model(prompt):
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 50            # Number of denoising steps
    guidance_scale = 15                # Scale for classifier-free guidance
    #generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
    batch_size = 1

    # Prep text 
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8)#,
    #generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            # Scale the latents (preconditioning):
            # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Display
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    now = datetime.now()
    fecha = str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'_at_'+str(now.hour)+':{0:02d}'.format(now.minute)
    print('We got a user!')
    try:
        f= open(fecha+".txt","w+")
        f.write('Generación individual')
        f.write(prompt)
        f.close()
    except:
        print('el directorio no se creo')
    return  pil_images[0]

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)# clamp set input into the range [ min, max ].
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def mod(prompt, imagen):
    width0, height0 = imagen.size
    input_image = imagen.resize((512, 512))
    encoded = pil_to_latent(input_image)
    # Settings (same as before except for the new prompt)
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 50            # Number of denoising steps
    guidance_scale = 9                  # Scale for classifier-free guidance
    #generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
    batch_size = 1

    # Prep text (same as before)
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler (setting the number of inference steps)
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents (noising appropriately for start_step)
    start_step = 15
    start_sigma = scheduler.sigmas[start_step]
    noise = torch.randn_like(encoded) # This the Guacamaya picture as the starting point
    latents = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
    latents = latents.to(torch_device).float()

    # Loop
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        if i > start_step: # << This is the only modification to the loop we do
            
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    now = datetime.now()
    fecha = str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'_at_'+str(now.hour)+':{0:02d}'.format(now.minute)
    print('We got a user!')
    try:
        f= open(fecha+".txt","w+")
        f.write('Generación individual')
        f.write(prompt)
        f.close()
    except:
        print('el directorio no se creo')
    return latents_to_pil(latents)[0].resize((width0, height0))

def gen_publi(text):
    return eval_model(text)
    
def gen_mult_publi(text):
    return (eval_model(text), eval_model(text), eval_model(text), eval_model(text))

def modify(text, image):
    return mod(text, image)

with gr.Blocks() as interfaz:
    gr.Image('iconohugg-removebg.png')
    gr.Markdown(
    '''
    <p style="text-align: center;">Marketing and design made simple using AI.</p>
    
    ''')
  
    with gr.Tab("Generate"):
        gr.Markdown(
            '''
            <p style="text-align: center;">
            To start generating ads, write what is your product and the style you want in your ad
            e.g: 'elegant shoes in a modern style'
            </p>
            '''
            )
        gr.Markdown(
            '''
            
            '''
            )
        gr.Markdown(
            '''
            <p style="text-align: center;">
            Below is a text box where you can tell us your product.
            </p>
            '''
            )
        gr.Markdown(
            '''
            
            '''
        )
        gr.Markdown(
            '''
            <p style="text-align: center;">
            With the generate button, what you wrote will begin to be processed and an advertising image will be generated in the box below.
            </p>
            '''
            )
        with gr.Column():
            inp = gr.Textbox(label='What is your product?',placeholder="Describe what you sell and the style of the ads you want")
            gen = gr.Button("Generate", variant='primary')
        out = gr.Image()
        gen.click(gen_publi, inp, out)
    with gr.Tab("Mosaic"):
        gr.Markdown(
        '''
          
        '''
        )
        gr.Markdown(
            '''
            <p style="text-align: center;">
           You can also create a mosaic of ad options to choose the one you like best.
           </p>
            '''
            )
        gr.Markdown(
            '''
            
            '''
            )
        gr.Markdown(
            '''
            <p style="text-align: center;">
          Below is a text box where you can tell us your product.
          </p>
            '''
            )
        gr.Markdown(
            '''
            
            '''
        )
        gr.Markdown(
            '''
            <p style="text-align: center;">
          With the generate button, what you wrote will begin to be processed and a mosaic of advertising images will be generated in the         
           box below.
           </p>
            '''
            )
        with gr.Column():
            inp = gr.Textbox(label='What is your product?',placeholder="Describe us what you sell")
            gen = gr.Button("Generate mosaic", variant='primary')
        with gr.Column():
            with gr.Row():
                out1 = gr.Image()
                out2 = gr.Image()
            with gr.Row():
                out3 = gr.Image()
                out4 = gr.Image()
        gen.click(gen_mult_publi, inp, [out1, out2, out3, out4])
    with gr.Tab("Modify"):
        gr.Markdown(
        '''
<p style="text-align: center;">        WARNING: This functionality is under development, proper operation is not guaranteed.</p>
        '''
        )
        gr.Markdown(
        '''
          
        '''
        )
        gr.Markdown(
        
        '''
        <p style="text-align: center;">
        You can also make changes to advertising you have.</p>
        '''
        )
        gr.Markdown(
            '''
            
            '''
            )
        gr.Markdown(
            '''
            <p style="text-align: center;">
               Below is a text box in which modifications you want to make to the advertising image.</p>
            '''
            )
        gr.Markdown(
            '''
            
            '''
        )
        gr.Markdown(
            '''
                 <p style="text-align: center;">You will find a box where you can upload the image you want to modify.</p>
            '''
        )
        gr.Markdown(
            '''
            
            '''
        )
        gr.Markdown(
            '''
               <p style="text-align: center;">With the modify button, the program begins to make the changes.</p>
            '''
        )
           
        with gr.Column():
            inp1 = gr.Textbox(label='What advertising do you want to modify?', placeholder="Describe the modifications you want to make")
            inp2 = gr.Image(label='To which advertisement do you want to apply the modifications?', interactive=True,type="pil")
            gen = gr.Button("Modify", variant='primary')
        out = gr.Image(type='filepath')
        gen.click(modify, [inp1, inp2], out)
    
    with gr.Tab("Share and Manage Campaigns"):
        gr.Markdown(
        '''
          <p style="text-align: center;">WARNING: This functionality is under development, proper operation is not guaranteed.</p>
        '''
        )
        gr.Markdown(
        '''
          
        '''
        )
        gr.Markdown(
            '''
            <p style="text-align: center;">You can upload the advertising to your social networks, for this you must have an account with us.</p>
            '''
            )
        gr.Markdown(
            '''
            
            '''
            )
        gr.Markdown(
            '''
           <p style="text-align: center;">Below are two text boxes where you must enter your username and password.</p>
            
            '''
            )
        gr.Markdown(
            '''
            
            '''
        )
        gr.Markdown(
            '''
                <p style="text-align: center;">We will ask you to link the account with google so that we can access and share the advertising you want on your social networks.</p>
            '''
            )
        with gr.Row():
            with gr.Column(visible = False):
                btn1 = gr.Button("Load")

            with gr.Column():
                gr.Markdown(
                '''
                
                  <p style="font-weight: bold; "> Login to MatketINK marketing campaing management.</p>
                ''')
                inp1 = gr.Textbox(label='User',placeholder="Enter your user")
                inp2 = gr.Textbox(label='Password',placeholder="Enter your password")
                gen1 = gr.Button("Login", variant='primary')
                gen2 = gr.Button("Login with Google Ads", variant='primary')
                gen3 = gr.Button("Login with Instagram", variant='primary')
                gen4 = gr.Button("Login with Facebook Ads", variant='primary')
                
            with gr.Column(visible = False):
                btn2 = gr.Button("Load")
interfaz.launch()