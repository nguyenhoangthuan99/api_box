import json

with open('wrong_image/log_wrong_image.json', 'r') as fp:
    try:
        wrong_image = json.load(fp)
    except:
        wrong_image = {}
        print(wrong_image)