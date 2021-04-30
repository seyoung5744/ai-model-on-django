from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Post # models에 정의한 Post class를 불러온다

admin.site.register(Post) # admin에 model에 정의한 Post class를 추가시킨다