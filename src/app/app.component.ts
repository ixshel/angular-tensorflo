import { Component, OnInit } from '@angular/core';
import { from, defer, animationFrameScheduler, timer, of, Observable } from 'rxjs';
import { concatMap, tap, map, observeOn, takeUntil, repeat } from 'rxjs/operators';

// import tf model <COCO-SSD>
import * as cocoSSD from '@tensorflow-models/coco-ssd';
import * as mobileNET from '@tensorflow-models/mobilenet';
import * as posenet from '@tensorflow-models/posenet';
import { SubSink } from 'subsink';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements OnInit {
  private subs = new SubSink();

  title = 'TF-ObjectDetection';
  private video: HTMLVideoElement;
  imageObj = new Image();
  imageName = 'https://angular.io/assets/images/logos/angular/angular.png';

  constructor() {
    this.imageObj.src = this.imageName;
  }

  ngOnInit() {
    this.webcam_init();
    this.predictWithModel();
  }

  public async predictWithModel() {
    // defaulted to lite_mobilenet_v2
    const model = await cocoSSD.load();
    // const model = await mobileNET.load();
    this.detectFrame(this.video, model);

    // await this.video.addEventListener('loadeddata', (async () => {
    //   console.log('loaded data');
    //   // Posenet
    //   const action$ = (model: posenet.PoseNet) =>
    //     defer(() => model.estimateMultiplePoses(this.video)).pipe(
    //       observeOn(animationFrameScheduler),
    //       tap((predictions: posenet.Pose[]) => this.renderPosenetPredictions(predictions)),
    //       takeUntil(timer(1000)),
    //       repeat()
    //     );

    //   this.subs.add(
    //     from(posenet.load({
    //       architecture: 'ResNet50',
    //       outputStride: 32,
    //       inputResolution: 257,
    //       quantBytes: 2
    //     })).pipe(
    //       concatMap(model => this.loadImage$().pipe(map(() => model))),
    //       concatMap(model => action$(model)),
    //     ).subscribe()
    //   );
    // }));


  }

  webcam_init() {
    this.video = <HTMLVideoElement>document.getElementById("video");

    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
        }
      })
      .then(stream => {
        this.video.srcObject = stream;
        this.video.onloadedmetadata = () => {
          this.video.play();
        };
      });
  }

  detectFrame = async (video, model) => {

    // Predictive model for images or video
    await model.detect(video).then(predictions => {
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
    });
  }

  renderPredictions = predictions => {
    const canvas = <HTMLCanvasElement>document.getElementById("canvas");

    const ctx = canvas.getContext("2d");

    canvas.width = 1000;
    canvas.height = 850;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    ctx.drawImage(this.video, 0, 0, 1000, 850);

    let i: number = 0;
    predictions.forEach(prediction => {
      i++;
      const x = prediction.bbox[0] + 210;
      const y = prediction.bbox[1] + 70;
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // Draw the bounding box.

      if (i % 2 === 1) {
        // odd
        ctx.strokeStyle = "#00FFFF";
      } else {
        ctx.strokeStyle = "#f90";
      }

      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach(prediction => {
      const x = prediction.bbox[0] + 210;
      const y = prediction.bbox[1] + 70;
      // Draw the text last to ensure it's on top.
      if (i % 2 === 1) {
        // odd
        // ctx.strokeStyle = "#00FFFF";
        ctx.fillStyle = "#000000";
      } else {
        ctx.fillStyle = "#000600";
      }

      ctx.fillText(prediction.class, x, y);
    });
  };

  loadImage$(): Observable<(observer: any) => void> {
    return of((observer: any) => {
      this.imageObj.onload = () => {
        observer.onNext(this.imageObj);
        observer.onCompleted();
      };
      this.imageObj.onerror = (err) => {
        observer.onError(err);
      };
    });
  }

  async detectPosenet(video, model) {
    await video.addEventListener('loadeddata', (async () => {
      console.log('loaded data');
      await model.estimateMultiplePoses(video)
        .then(res => {
          // console.log('Res: ', res);
          this.renderPosenetPredictions(res);
        })
        .catch(error => {
          console.log('Error: ', error);
        })
    }));

  }

  renderPosenetPredictions(k) {
    // console.log('data: ', k);
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');

    canvas.width = 800;
    canvas.height = 600;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);

    const minConfidence = 0.15;

    k.forEach(({ score, keypoints }) => {
      // console.log('kpts', keypoints);
      this.drawKeypoints(keypoints, minConfidence, ctx);
      this.drawSkeleton(keypoints, minConfidence, ctx);
    });
  }

  drawKeypoints(keypoints: posenet.Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1) {
    for (let i = 0; i < keypoints.length; i++) {
      const keypoint = keypoints[i];
      if (keypoint.score < minConfidence) {
        continue;
      }
      const { y, x } = keypoint.position;
      if (i > 5) {
        this.drawPoint(ctx, y * scale, x * scale, 3, 'aqua');
      } else {
        if (i === 0) {
          const qrcodeSize = this.calDistance(keypoints[3].position, keypoints[4].position) * 2;
          ctx.drawImage(this.imageObj, x - (qrcodeSize / 2), y - (qrcodeSize / 2), qrcodeSize, qrcodeSize);
        }
      }
    }
  }

  drawPoint(ctx: CanvasRenderingContext2D, y: number, x: number, r: number, color: string) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }

  calDistance(position1: { x: number, y: number }, position2: { x: number, y: number }, ): number {
    return Math.floor(Math.sqrt((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2));
  }
  toTuple({ y, x }) {
    return [y, x];
  }

  drawSkeleton(keypoints: posenet.Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1) {
    const adjacentKeyPoints =
      posenet.getAdjacentKeyPoints(keypoints, minConfidence);
    adjacentKeyPoints.forEach((keypoint) => {
      this.drawSegment(
        this.toTuple(keypoint[0].position), this.toTuple(keypoint[1].position), 'aqua',
        scale, ctx);
    });
  }

  drawSegment([ay, ax]: any, [by, bx]: any, color: string, scale: number, ctx: CanvasRenderingContext2D) {
    ctx.beginPath();
    ctx.moveTo(ax * scale, ay * scale);
    ctx.lineTo(bx * scale, by * scale);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
  }

}