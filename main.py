from flask import Flask
from app import views

app=Flask(__name__)

app.add_url_rule('/','base',views.base)
app.add_url_rule('/faceapp','faceapp',views.faceapp,methods=['GET','POST'])



if __name__=="__main__":
    app.run()