from django.conf.urls import url
from yolov4 import views
from yolov4.apps import Yolov4Config
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path


# 어떤 글자가 들어오던지 views의 first_view 함수로 가라
# /uimage 로 들어오면 views의 uimage 함수로 전달해라
urlpatterns = [
    # url(r'^$', views.first_view, name='first_view'),
    path('',views.first_view),
    url(r'^uimage/$', views.uimage, name='uimage'),
    # url(r'^dface/$', views.dface, name='dface'),
    path('dface/',views.dface, name='dface'),
]

# 이미지 파일을 업로드 하기 위한 설정
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)