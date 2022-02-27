from django.contrib import admin
from django.urls import path,include
from django.urls import re_path as url
from resume_prediction_frontpage import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    #path('', views.home, name='home'),
    #path('', views.index, name='index'),
    path('admin/', admin.site.urls),
    #path('signup', views.signup, name='signup'),
    #path('activate/<uidb64>/<token>', views.activate, name='activate'),
    #path('signin', views.signin, name='signin'),
    #path('signout', views.signout, name='signout'),
    #url('^$',views.index,name='Homepage'),
    #path('get_resume', views.showfile, name='fileupload1'),
    #path('upload', views.UploadView.as_view(), name='fileupload'),
    path('', views.home, name='home'),
    path('home', views.home, name='home'),
    path('aboutus', views.aboutus, name='aboutus'),
    path('upload/', views.upload, name='upload'),
    path('delete/multiple_uploads', views.multiple_uploads, name='delete'),
    path('multiple_uploads/', views.multiple_uploads, name='multiple uploads'),
    path("solutions", views.solutions, name='solutions'),
    path("predict", views.predict, name='predict'),
    path("predict_all", views.predict_all, name='predict_all'),
    path("visualize", views.visualize, name='visualize'),
    path("contact", views.contact, name='contact'),
]+static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)