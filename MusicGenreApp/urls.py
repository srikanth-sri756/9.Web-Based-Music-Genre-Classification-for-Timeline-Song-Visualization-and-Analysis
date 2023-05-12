from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
			path("Login.html", views.Login, name="Login"),
			path("LoginAction", views.LoginAction, name="LoginAction"),
			path("Signup.html", views.Signup, name="Signup"),
			path("SignupAction", views.SignupAction, name="SignupAction"),	    	
			path("TrainSVM", views.TrainSVM, name="TrainSVM"),
			path("TrainDT", views.TrainDT, name="TrainDT"),
			path("TrainLSTM", views.TrainLSTM, name="TrainLSTM"),
			path("TrainFF", views.TrainFF, name="TrainFF"),
			path("Classification", views.Classification, name="Classification"),
			path("ClassificationAction", views.ClassificationAction, name="ClassificationAction"),
]