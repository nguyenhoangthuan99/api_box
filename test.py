import pickle, json



with open('login.pickle', 'rb') as handle:
    fake_users_db = pickle.load(handle)
print(fake_users_db)
def d(x1,y1,x2,y2):
   return ((x1-x2)**2  +(y1-y2)**2)**0.5

with open('list_cam.pickle', 'rb') as handle:
    list_camera = pickle.load(handle)

config = {"users":fake_users_db,
"cameras": list_camera
}
with open('config.json', 'w') as fp:
    json.dump(config, fp)


with open('config.json', 'r') as fp:
    config = json.load(fp)

print(mapping==config)
