import numpy as np
from PIL import Image

def test_initialize_model():
    from main import initialize_model

    model_list = ['saycam', 'saycamA', 'saycamS', 'saycamY', 'resnet50', 'clipRN50', 'clipRN50x4',
                  'clipRN50x16', 'clipViTB32', 'clipViTB16', 'dino_resnet50', 'alexnet', 'vgg16',
                  'swav', 'mocov2']

    embedding_sizes = [2048, 2048, 2048, 2048, 2048, 1024, 640, 768, 512, 512, 2048, 4096, 4096, 2048, 2048]
    fake_image = np.uint8(np.random.randint(0, 255, (3, 224, 224)))
    fake_image = Image.fromarray(fake_image, 'RGB')

    for i in range(len(model_list)):
        model_type = model_list[i]
        embedding_size = embedding_sizes[i]

        model, penult_model, transform = initialize_model(model_type)
        fake_image_t = transform(fake_image).unsqueeze(0)

        if model_type == 'clipRN50' or model_type == 'clipViTB32' or model_type == 'clipRN50x4' \
                or model_type == 'clipRN50x16' or model_type == 'clipViTB16':
            embedding = penult_model.encode_image(fake_image_t).detach().numpy().squeeze()
        else:
            embedding = penult_model(fake_image_t).detach().numpy().squeeze()

        print("Running test for {0}...".format(model_type))
        assert embedding_size == embedding.shape[0]

