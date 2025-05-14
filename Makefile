init:
	docker-compose up --build -d
	
py:
	docker-compose exec -it app /bin/bash

start:
	docker-compose up -d
	docker-compose exec app python3 /app/app/main.py

video:
	docker-compose exec app python3 /app/app/video.py --image_path "backup/Mecha/sagittaire_**_robot_mecha_style_1747123323_final.png" --output_path "video_output.mp4" --use_4bit