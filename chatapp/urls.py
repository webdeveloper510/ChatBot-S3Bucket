from django.urls import path
from . import views
urlpatterns = [
      path('', views.getting_details),
]