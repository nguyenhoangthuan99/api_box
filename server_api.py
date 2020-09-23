from fastapi import Depends, FastAPI, HTTPException, status
 
app = FastAPI()
new_version = "1"
date = "12/1/2020"
link = "https://"
description = ""
size = ""
@app.post("/checkUpdate")
def checkUpdate(username:str,password:str,version:str):
    if new_version == version:
        return {"result": "there is no new version" }, 200
    else:
        return {"version": new_version,"date":date, "link":link, "description":description,"size": size}
    
