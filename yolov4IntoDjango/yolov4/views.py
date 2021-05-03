from django.shortcuts import render
from django.shortcuts import redirect
from .forms import UploadImageForm, ImageUploadForm # 이미지 업로드 form
from django.core.files.storage import FileSystemStorage # 이미지 저장
from django.conf import settings
# from .opencv_dface import opencv_dface
from yolov4.detect import detect

# Create your views here.

# urls에서 요청이 오면 first_view로 오는데, 이 때 ~.html을 rendering 해줘라.
def first_view(request):
    return render(request, "yolov4/first_view.html",{})

def uimage(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)  # 이미지르 업로드할때 쓰는 form
        if form.is_valid():
            myfile = request.FILES['image']
            fs = FileSystemStorage()  # 이미지 파일을 저장할때 쓰는 함수
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            return render(request, 'yolov4/uimage.html', {'form': form, 'uploaded_file_url' : uploaded_file_url})
    else:
        form = UploadImageForm()
        return render(request, 'yolov4/uimage.html', {'form': form})


def dface(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if request.method == 'POST':
        if form.is_valid():
            post = form.save(commit=False)
            post.save()

            imageURL = settings.MEDIA_URL + form.instance.document.name
            print("경로:",settings.MEDIA_ROOT_URL + imageURL)
            # opencv_dface(settings.MEDIA_ROOT_URL + imageURL)
            detect(settings.MEDIA_ROOT_URL + imageURL)
            return render(request, 'yolov4/dface.html', {'form': form, 'post': post})
    else:
        form = ImageUploadForm()
    return render(request, 'yolov4/dface.html', {'form': form})