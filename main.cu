//                         _ooOoo_
//                        o8888888o
//                        88" . "88
//                        (| -_- |)
//                        O\  =  /O
//                     ____/'---'\____
//                   .'  \\|     |//  '.
//                  /  \\|||  :  |||//  \
//                 /  _||||| -:- |||||-  \
//                 |   | \\\  _  /// |   |
//                 | \_|  ''\---/''  |   |
//                 \  .-\__  '_'  ___/-. /
//               ___`. .'  /--.--\  '. . __
//            ."" '<  `.___\_<|>_/___.'  >'"".
//           | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//           \  \ `-.   \_ __\ /__ _/   .-`  / /
// ===========`-.____`-.___\_____/___.-`_____.-`================
//                          `=---='
// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// =============================================================

//152行不能整除，整除出bug
//kernel_size不能超过block_size_x和block_size_y，不然会出现越界的问题
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <malloc.h>
/*//#include <helper_gl.h>
#include <helper_cuda.h>
//#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <GL/freeglut.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glx.h>*/

#define uint unsigned int
#define uchar unsigned char
using namespace std;
using namespace cv;
#define block_size_x 8
#define block_size_y 64

//new define sum area
//kernel_size>=2*box_kernel+1!!
#define box_kernel 3
#define MIDDLE_kernel 0.09//调节加权中值滤波取的第几个值,配合sigma_c
#define MIDDLE_kernel_h 0.8
#define error_range 25
#define Middle_Radius 3
#define filterweight_kernel 3//size = weightedmedian_kernel*2+1
#define filterweight_kernel_h 2//size = weightedmedian_kernel*2+1
#define kernel_size box_kernel*2+1
#define min(x,y)  ( x<y?x:y )
#define max(x,y)  ( x<y?y:x )

#define rows 992   //948*1500
#define cols 1420
#define disp_max 160
#define scale_pic 1
#define sigma_c 8//weightedmedianfilter parameter
#define sigma_p 16
#define r1 5
#define step 1

uchar (*temp)[cols];

//int shift=(kernel_size+1)/2;

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void Match_error(uchar (*a)[cols],uchar (*b)[cols],bool (*c)[cols])
{
	const uint idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	if((idy>disp_max+(kernel_size-1)/2+box_kernel)&&(idy<cols-disp_max-(kernel_size-1)/2-box_kernel)&&(idx>(kernel_size-1)/2+box_kernel)&&(idx<rows-(kernel_size-1)/2-box_kernel))
	{
		int a0=a[idx][idy];
		uint yy=idy-a0;
		int b0=b[idx][yy];
		c[idx][idy]=((a0-b0)<error_range)?0:1;
	}
//	a[idx][idy]=(uchar)a[idx][idy];
}

__global__ void near_match(	uchar (*a)[cols],uchar (*a1)[cols],bool (*b)[cols],uchar (*r)[cols],
							uchar (*l)[cols],uchar (*u)[cols],uchar (*d)[cols],
							uchar (*lr)[cols],uchar (*ud)[cols])
//a是视察图，b是匹配错误分析图
{
	const uint idx=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint idy=(blockIdx.y*blockDim.y)+threadIdx.y;
	r[idx][idy]=0;
	l[idx][idy]=0;
	u[idx][idy]=0;
	d[idx][idy]=0;
	lr[idx][idy]=0;
	ud[idx][idy]=0;
	a1[idx][idy]=a[idx][idy];
	int win_move=(r1-1)/2+1;
	uint k;
	for (k=0; k<win_move; k++)
	{
		if(b[idx][idy+k]==0&(idy+k)<cols)
			{
			r[idx][idy]=a[idx][idy+k];
			break;
			}
	}
	for (k=0; k<win_move; k++)
	{
		int T=idy-k;
		if(b[idx][T]==0&(T>0))
		{
			l[idx][idy]=a[idx][T];
			break;
		}
	}
	for (k=0; k<win_move; k++)
	{
		int T=idx-k;
		if(b[T][idy]==0 &(T)>0)
		{
			u[idx][idy]=a[T][idy];
			break;
		}
	}

	for (k=0; k<win_move; k++)
	{
		if(b[idx+k][idy]==0 &(idx+k)<cols)
		{
			d[idx][idy]=a[idx+k][idy];
			break;
		}
	}
	if(b[idx][idy]==1)
	{
		if ((r[idx][idy]==0)&&(l[idx][idy]>0))
		{
			lr[idx][idy]=l[idx][idy];
		}
		if ((r[idx][idy]>0)&&(l[idx][idy]==0))
		{
			lr[idx][idy]=r[idx][idy];
		}
		if ((r[idx][idy]>0)&&(l[idx][idy]>0))
		{
			lr[idx][idy]=min(r[idx][idy],l[idx][idy]);
		}

		if ((u[idx][idy]==0)&&(d[idx][idy]>0))
		{
			ud[idx][idy]=d[idx][idy];
		}
		if ((u[idx][idy]>0)&&(d[idx][idy]==0))
		{
			ud[idx][idy]=u[idx][idy];
		}
		if ((u[idx][idy]>0)&&(d[idx][idy]>0))
		{
			ud[idx][idy]=min(u[idx][idy],d[idx][idy]);
		}
		if ((ud[idx][idy]>0)&&(lr[idx][idy]==0))
		{
			a1[idx][idy]=ud[idx][idy];
			b[idx][idy]=0;
		}
		if ((ud[idx][idy]==0)&&(lr[idx][idy]>0))
		{
			a1[idx][idy]=lr[idx][idy];
			b[idx][idy]=0;
		}
		if ((ud[idx][idy]>0)&&(lr[idx][idy]>0))
		{
			a1[idx][idy]=min(lr[idx][idy],ud[idx][idy]);
			b[idx][idy]=0;
		}
	}
}

/*
 __global__ void stereo_kernel(uint (*a)[cols],uint (*b)[cols],uchar (*disp)[cols])
{
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	__shared__ uint sh_left[block_size_x][block_size_y];
	__shared__ uint sh_right[block_size_x][block_size_y+disp_max];
	int y1=blockIdx.y*blockDim.y+threadIdx.y-disp_max;
	int y2=blockIdx.y*blockDim.y+threadIdx.y+64-disp_max;
	int y3=blockIdx.y*blockDim.y+threadIdx.y+128-disp_max;
	if(y<disp_max)
	{
		disp[x][y]=0;
	}
	else
	{
		sh_left[threadIdx.x][threadIdx.y]=a[x][y];
		if(threadIdx.y<32)
		{
			sh_right[threadIdx.x][threadIdx.y]         =b[x][y1];
			sh_right[threadIdx.x][threadIdx.y+64]      =b[x][y2];
			sh_right[threadIdx.x][threadIdx.y+128]     =b[x][y3];
			sh_right[threadIdx.x][threadIdx.y+disp_max]=b[x][y];
		}
		else
		{
			sh_right[threadIdx.x][threadIdx.y]         =b[x][y1];
			sh_right[threadIdx.x][threadIdx.y+64]      =b[x][y2];
			sh_right[threadIdx.x][threadIdx.y+disp_max]=b[x][y];
		}
		__syncthreads();
		disp[x][y]=0;
		uint cost=abs((float)(sh_left[threadIdx.x][threadIdx.y]-sh_right[threadIdx.x][threadIdx.y+disp_max])); //(float)(b[x][y])
		uint cost_now;
		for(int d=1;d<disp_max+1;d+=step)
		{
			cost_now=abs((float)(sh_left[threadIdx.x][threadIdx.y]-sh_right[threadIdx.x][threadIdx.y+(disp_max-d)]));  //(float)(b[x][y-d])
			if(cost>cost_now)
			{
				disp[x][y]=d;
				cost=cost_now;
			}
		}
	}
}
 */

__global__ void stereo_kernel(uint (*a)[cols],uint (*b)[cols],uchar (*disp)[cols])
{
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	//test0=10;
	//因为用的是3X3的区域内计算，所以上下左右方向各扩充一格

	__shared__ uint sh_left[block_size_x+kernel_size-1+box_kernel*2+1][block_size_y+kernel_size-1+box_kernel*2+1];//共享内存为什么要大点，共享内存只能在方格内部起作用
	__shared__ uint sh_right[block_size_x+kernel_size-1+box_kernel*2+1][block_size_y+disp_max+kernel_size-1+box_kernel*2+1];

	//这些int都相当于寄存器
	//int y1=blockIdx.y*blockDim.y+threadIdx.y-disp_max;//这个定义中64,128还有就是disp_max作用？0-64
	//int y2=blockIdx.y*blockDim.y+threadIdx.y+64-disp_max;//block_size_y的大小是64,视差大小是160,所以要3个64，64-128
	//int y3=blockIdx.y*blockDim.y+threadIdx.y+128-disp_max;//128-192

	if((y>disp_max+(kernel_size-1)/2+box_kernel)&&(y<cols-(kernel_size-1)/2-box_kernel)&&(x>(kernel_size-1)/2+box_kernel)&&(x<rows-(kernel_size-1)/2-box_kernel))//y>disp_max，表示了左边disp_max个点不要了
	{
		//configure left shared memory
		sh_left[threadIdx.x][threadIdx.y]=a[x-(kernel_size-1)/2][y-(kernel_size-1)/2];//因为共享内存上下左右各大了一行，而坐标原点在左上角，所以需要全部-1
		if(threadIdx.x<(kernel_size-1+box_kernel))//内存重复覆盖的问题？这里这个threadIdx.x是不是一定小于2,因为block_size_x是2，所以判断语句是不是不需要了？
		{
			sh_left[threadIdx.x+block_size_x][threadIdx.y]=a[x+block_size_x-(kernel_size-1)/2][y-(kernel_size-1)/2];//相当于存了4行的数据，上下各拓宽了一行
		}
		if(threadIdx.y<(kernel_size-1+box_kernel))//这个是为了扩充右边界，因为上下左右各拓宽了一行
		{
			sh_left[threadIdx.x][threadIdx.y+block_size_y]=a[x-(kernel_size-1)/2][y+block_size_y-(kernel_size-1)/2];
		}
		if((threadIdx.x<(kernel_size-1+box_kernel))&&(threadIdx.y<(kernel_size-1+box_kernel)))//这个是为了更新边框右下角的数据
		{
			sh_left[threadIdx.x+block_size_x][threadIdx.y+block_size_y]=a[x+block_size_x-(kernel_size-1)/2][y+block_size_y-(kernel_size-1)/2];
		}

		//configure right shared memory
		//y和x不同，y需要考虑视差160的因素，所以需要3个blocks才能计算一个视差出来，如果恰巧在边界上应该计算4个blocks？
		int cacl_number=disp_max-block_size_y*(disp_max/block_size_y);//这个地方不能整除，整除出bug
		int cacl_Num=(disp_max+block_size_y-1)/block_size_y;
		if(threadIdx.y<(cacl_number))//在0-32的范围内，64×3=32+160,正好只需要3个blocks就能够完成计算
		{
			for(int i=0;i<cacl_Num;i++)
			{
				sh_right[threadIdx.x][threadIdx.y+block_size_y*i] = b[x-(kernel_size-1)/2][y-disp_max-(kernel_size-1)/2+block_size_y*i];
			}
			sh_right[threadIdx.x][threadIdx.y+disp_max] = b[x-(kernel_size-1)/2][y-(kernel_size-1)/2];
			//sh_right[threadIdx.x][threadIdx.y]         =b[x-(kernel_size-1)/2][y1-(kernel_size-1)/2];//正常的第一个blocks0-64，这里是0-31的视差
			//sh_right[threadIdx.x][threadIdx.y+64]      =b[x-1][y2-1];//第二个blocks64-128，这里是64-95的视差
			//sh_right[threadIdx.x][threadIdx.y+128]     =b[x-1][y3-1];//第三个blocks128-192，这里是128-159的视差
			//sh_right[threadIdx.x][threadIdx.y+disp_max]=b[x-1][y-1];//这里是160-191的视差范围，这些点虽然不算，但是计算时需要用到
			if(threadIdx.x<(kernel_size-1))//配置下边界的数据
			{
				for(int i=0;i<cacl_Num;i++)
				{
					sh_right[threadIdx.x+block_size_x][threadIdx.y+block_size_y*i] = b[x+block_size_x-(kernel_size-1)/2][y-disp_max-(kernel_size-1)/2+block_size_y*i];
				}
				sh_right[threadIdx.x+block_size_x][threadIdx.y+disp_max]=b[x+block_size_x-(kernel_size-1)/2][y-(kernel_size-1)/2];
				//sh_right[threadIdx.x+block_size_x][threadIdx.y]         =b[x+block_size_x-(kernel_size-1)/2][y1-(kernel_size-1)/2];
				//sh_right[threadIdx.x+2][threadIdx.y+64]      =b[x+1][y2-1];
				//sh_right[threadIdx.x+2][threadIdx.y+128]     =b[x+1][y3-1];
				//sh_right[threadIdx.x+2][threadIdx.y+disp_max]=b[x+1][y-1];
			}
			if(threadIdx.y<(kernel_size-1))//配置右边界数据，这里的224是64+160,这个if的位置有所改动
			{
				sh_right[threadIdx.x][threadIdx.y+block_size_y+disp_max]=b[x-(kernel_size-1)/2][y+block_size_y-(kernel_size-1)/2];//这里实在是太牛逼了，
			}//																			//blockIdx.y*blockDim.y相当于前一个block的最后一个数据
			if((threadIdx.x<(kernel_size-1))&&(threadIdx.y<(kernel_size-1)))//这里配置右下角
			{
				sh_right[threadIdx.x+block_size_x][threadIdx.y+block_size_y+disp_max]=b[x+block_size_x-(kernel_size-1)/2][y+block_size_y-(kernel_size-1)/2];//这个地方还是有个问题，将一个b值付给了2个thread值
			}
		}
		else//当threadIdx.y>32时，需要4个blocks才能完成计算，因为64*3<threadIdx.y+160
		{
			for(int i=0;i<(cacl_Num-1);i++)
			{
				sh_right[threadIdx.x][threadIdx.y+block_size_y*i] = b[x-(kernel_size-1)/2][y-disp_max-(kernel_size-1)/2+block_size_y*i];
			}
			sh_right[threadIdx.x][threadIdx.y+disp_max] = b[x-(kernel_size-1)/2][y-(kernel_size-1)/2];
			//sh_right[threadIdx.x][threadIdx.y]         =b[x-1][y1-1];//这里64>threadIdx.y>32的视差
			//sh_right[threadIdx.x][threadIdx.y+64]      =b[x-1][y2-1];//这里是96-127的视差
			//sh_right[threadIdx.x][threadIdx.y+disp_max]=b[x-1][y-1];//这里是192-223的视差
			if(threadIdx.x<(kernel_size-1))//配置对应的下边界
			{
				for(int i=0;i<(cacl_Num-1);i++)
				{
					sh_right[threadIdx.x+block_size_x][threadIdx.y+block_size_y*i] = b[x+block_size_x-(kernel_size-1)/2][y-disp_max-(kernel_size-1)/2+block_size_y*i];
				}
				sh_right[threadIdx.x+block_size_x][threadIdx.y+disp_max]=b[x+block_size_x-(kernel_size-1)/2][y-(kernel_size-1)/2];
				//sh_right[threadIdx.x+2][threadIdx.y]         =b[x+1][y1-1];
				//sh_right[threadIdx.x+2][threadIdx.y+64]      =b[x+1][y2-1];
				//sh_right[threadIdx.x+2][threadIdx.y+disp_max]=b[x+1][y-1];
			}
			if(threadIdx.y<(kernel_size-1))//配置对应的右边界
			{
				sh_right[threadIdx.x][threadIdx.y+block_size_y+disp_max]=b[x-(kernel_size-1)/2][y+block_size_y-(kernel_size-1)/2];
				//sh_right[threadIdx.x][threadIdx.y+224]=b[x-1][blockIdx.y*blockDim.y+63];
			}
			if((threadIdx.x<(kernel_size-1))&&(threadIdx.y<(kernel_size-1)))//配置右下边界
			{
				sh_right[threadIdx.x+block_size_x][threadIdx.y+block_size_y+disp_max]=b[x+block_size_x-(kernel_size-1)/2][y+block_size_y-(kernel_size-1)/2];
				//sh_right[threadIdx.x+2][threadIdx.y+224]=b[x+1][blockIdx.y*blockDim.y+63];
			}
		}
		// temp1存0-8位，temp2存9-15位，temp3存16-24位
		int model1=255;
		int model2=255*2^8;
		int temp1_l=0;
		int temp1_r=0;
		int temp2_l=0;
		int temp2_r=0;
		int temp3_l=0;
		int temp3_r=0;
		__syncthreads();
		disp[x][y]=0;
		int cost=0;
		for(int i=(-box_kernel);i<(box_kernel+1);i++)
		{
//		int i =0;
			for (int j=(-box_kernel);j<(box_kernel+1);j++)
			{
				temp1_l=((int)sh_left[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+(kernel_size-1)/2+j])&model1;
				temp1_r=((int)sh_right[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+disp_max+(kernel_size-1)/2+j])&model1;
				temp2_l=((int)sh_left[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+(kernel_size-1)/2+j])&model2;
				temp2_l=temp2_l>>8;
				temp2_r=((int)sh_right[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+disp_max+(kernel_size-1)/2+j])&model2;
				temp2_r=temp2_r>>8;
				temp3_l=((int)sh_left[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+(kernel_size-1)/2+j])>>16;
				temp3_r=((int)sh_right[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+disp_max+(kernel_size-1)/2+j])>>16;
				cost+=abs(temp1_l-temp1_r)+abs(temp2_l-temp2_r)+abs(temp3_l-temp3_r);
//				cost+=abs((int)sh_left[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+(kernel_size-1)/2+j]-(int)sh_right[threadIdx.x+(kernel_size-1)/2+i][threadIdx.y+disp_max+(kernel_size-1)/2+j]);
			}
		}
		//
//		cost=		(//abs((float)(sh_left[threadIdx.x][threadIdx.y]-sh_right[threadIdx.x][threadIdx.y+disp_max]))
//						+abs((float)sh_left[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(kernel_size-1)/2]-(float)sh_right[threadIdx.x+(kernel_size-1)/2][threadIdx.y+disp_max+(kernel_size-1)/2])
						//+abs((float)(sh_left[threadIdx.x][threadIdx.y+2]-sh_right[threadIdx.x][threadIdx.y+disp_max+2]))
					//	+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2-1][threadIdx.y+(kernel_size-1)/2]-sh_right[threadIdx.x+(kernel_size-1)/2-1][threadIdx.y+disp_max+(kernel_size-1)/2]))
					//	+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2+1][threadIdx.y+(kernel_size-1)/2]-sh_right[threadIdx.x+(kernel_size-1)/2+1][threadIdx.y+disp_max+(kernel_size-1)/2]))
					//	+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(kernel_size-1)/2-1]-sh_right[threadIdx.x+(kernel_size-1)/2][threadIdx.y+disp_max+(kernel_size-1)/2-1]))
						//+abs((float)(sh_left[threadIdx.x+2][threadIdx.y]-sh_right[threadIdx.x+2][threadIdx.y+disp_max]))
					//	+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(kernel_size-1)/2+1]-sh_right[threadIdx.x+(kernel_size-1)/2][threadIdx.y+disp_max+(kernel_size-1)/2+1]))
//						/*+abs((float)(sh_left[threadIdx.x+2][threadIdx.y+2]-sh_right[threadIdx.x+2][threadIdx.y+disp_max+2]))*/)/9; //(float)(b[x][y])
		int cost_now;
		for(int d=1;d<disp_max+1;d+=step)
		{
			cost_now=0;

			for(int k=(-box_kernel);k<(box_kernel+1);k++)
			{
				for (int l=(-box_kernel);l<(box_kernel+1);l++)
				{
					temp1_l=((int)sh_left[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(kernel_size-1)/2+l])&model1;
					temp1_r=((int)sh_right[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(disp_max-d)+(kernel_size-1)/2+l])&model1;
					temp2_l=((int)sh_left[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(kernel_size-1)/2+l])&model2;
					temp2_l=temp2_l>>8;
					temp2_r=((int)sh_right[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(disp_max-d)+(kernel_size-1)/2+l])&model2;
					temp2_r=temp2_r>>8;
					temp3_l=((int)sh_left[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(kernel_size-1)/2+l])>>16;
					temp3_r=((int)sh_right[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(disp_max-d)+(kernel_size-1)/2+l])>>16;
					cost_now+=abs(temp1_l-temp1_r)+abs(temp2_l-temp2_r)+abs(temp3_l-temp3_r);
//					cost_now+=abs((int)sh_left[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(kernel_size-1)/2+l]-(int)sh_right[threadIdx.x+(kernel_size-1)/2+k][threadIdx.y+(disp_max-d)+(kernel_size-1)/2+l]);
				}
			}
//			for (int l=(-box_kernel);l<(box_kernel+1);l++)
//			{
//				cost_now+=abs(((int)sh_left[threadIdx.x+(kernel_size-1)/2-4][threadIdx.y+(kernel_size-1)/2+l]-(int)sh_right[threadIdx.x+(kernel_size-1)/2-4][threadIdx.y+(disp_max-d)+(kernel_size-1)/2+l]);
//			}
			//
//			cost_now=		(//abs((float)(sh_left[threadIdx.x][threadIdx.y]-sh_right[threadIdx.x][threadIdx.y+(disp_max-d)]))
//							+abs((float)sh_left[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(kernel_size-1)/2]-(float)sh_right[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(disp_max-d)+(kernel_size-1)/2])
							//+abs((float)(sh_left[threadIdx.x][threadIdx.y+2]-sh_right[threadIdx.x][threadIdx.y+(disp_max-d)+2]))
							//+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2-1][threadIdx.y+(kernel_size-1)/2]-sh_right[threadIdx.x+(kernel_size-1)/2-1][threadIdx.y+(disp_max-d)+(kernel_size-1)/2]))
						//	+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2+1][threadIdx.y+(kernel_size-1)/2]-sh_right[threadIdx.x+(kernel_size-1)/2+1][threadIdx.y+(disp_max-d)+(kernel_size-1)/2]))
							//+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(kernel_size-1)/2-1]-sh_right[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(disp_max-d)+(kernel_size-1)/2-1]))
							//+abs((float)(sh_left[threadIdx.x+2][threadIdx.y]-sh_right[threadIdx.x+2][threadIdx.y+(disp_max-d)]))
							//+abs((float)(sh_left[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(kernel_size-1)/2+1]-sh_right[threadIdx.x+(kernel_size-1)/2][threadIdx.y+(disp_max-d)+(kernel_size-1)/2+1]))
//							/*+abs((float)(sh_left[threadIdx.x+2][threadIdx.y+2]-sh_right[threadIdx.x+2][threadIdx.y+(disp_max-d)+2]))*/)/9;  //(float)(b[x][y-d])
			if(cost>cost_now)
			{
				disp[x][y]=d;
				cost=cost_now;
			}
		}
	}
	else
	{
		disp[x][y]=0;
	}
}

__global__ void box_x(uchar (*input)[cols],uchar (*output)[cols],int win_radius)
{
	const uint idx= (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	uint scale=(win_radius<<1)+1;
	if ((idx >= win_radius) && (idx < rows - 1 - win_radius) && (idy >= win_radius) && (idy < cols - 1 - win_radius))
	{
		uint sum=0;
		for (int x = idx-win_radius; x <idx+win_radius+1 ; x++)
		{
			sum += input[x][idy];
		}
		output[idx][idy]=sum/scale;
	}
	else
		output[idx][idy]=input[idx][idy];
}

__global__ void box_y(uchar (*input)[cols],uchar (*output)[cols],int win_radius)
{
	const uint idx= (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy =(blockIdx.y*blockDim.y) + threadIdx.y;
	uint scale=(win_radius<<1)+1;
	if ((idx >= win_radius) && (idx < rows - 1 - win_radius) && (idy >= win_radius) && (idy < cols - 1 - win_radius))
	{
		uint sum=0;
		for (int y = idy-win_radius; y <idy+win_radius+1 ; y++)
		{
			sum += input[idx][y];
		}
		output[idx][idy]=sum/scale;
	}
	else
		output[idx][idy]=input[idx][idy];
}

__global__ void flip_kernel(uint (*input)[cols],uint (*output)[cols])
{
	const uint idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	if(idy<cols)
	output[idx][idy]=input[idx][cols-1-idy];
}
__global__ void flip_kernel_char(uchar (*input)[cols],uchar (*output)[cols])
{
	const uint idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	if(idy<cols)
	output[idx][idy]=input[idx][cols-1-idy];
}

void box_filter(uchar (*input)[cols],uchar (*output)[cols],int win_radius,dim3 grid_size,dim3 block_size)
{
	box_x<<<grid_size,block_size>>>(input,temp,win_radius);
	box_y<<<grid_size,block_size>>>(temp,output,win_radius);
}

__global__ void middle_filter(uchar (*input)[cols],uchar (*output)[cols],int middle_radius)
{
	const uint idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	int temp[Middle_Radius*Middle_Radius];
	int i;
	int j;
	if((idx>(middle_radius-1)/2)&&idx<(rows-(kernel_size-1)/2-1)&&idy<(cols-1-(kernel_size-1)/2-disp_max)&&idy>(disp_max+(kernel_size-1)/2))
	{
		for(int i=-(middle_radius-1)/2;i<(middle_radius+1)/2;i++)
		{
			for(int j=-(middle_radius-1)/2;j<(middle_radius+1)/2;j++)
			{
				temp[(i+(middle_radius-1)/2)*middle_radius+(j+(middle_radius-1)/2)]=input[idx+i][idy+j];
			}
		}

		for (int j=0; j<(middle_radius+1); ++j)
		    {
		        int min=j;
		        for (int l=j+1; l<middle_radius*middle_radius; ++l)
		            if (temp[l] < temp[min])
		                min=l;
		        const float bbtemp=temp[j];
		        temp[j]=temp[min];
		        temp[min]=bbtemp;
		    }
		output[idx][idy]= temp[(middle_radius-1)/2];
	}
	else
	{
		output[idx][idy]=0;
	}
}
__global__ void pic_show(uchar (*input)[cols],uchar (*output)[cols])
{
	const uint idx = (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	output[idx][idy]=input[idx][idy]*scale_pic;
}


__global__ void filter_weight(uint (*a)[cols],uchar (*b)[cols],uchar (*output)[cols],float (*temp_d)[2*filterweight_kernel+1])//filterweight_kernel
{
	//这里只要2*filterweight_kernel<block_size_x和block_size_y就行了
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	output[x][y]=0;
	//test0=10;
	//因为用的是3X3的区域内计算，所以上下左右方向各扩充一格

	__shared__ uint sh_pic[block_size_x+filterweight_kernel*2][block_size_y+filterweight_kernel*2];//共享内存为什么要大点，共享内存只能在方格内部起作用
	__shared__ uint sh_disp[block_size_x+filterweight_kernel*2][block_size_y+filterweight_kernel*2];
	//这些int都相当于寄存器
	//int y1=blockIdx.y*blockDim.y+threadIdx.y-disp_max;//这个定义中64,128还有就是disp_max作用？0-64
	//int y2=blockIdx.y*blockDim.y+threadIdx.y+64-disp_max;//block_size_y的大小是64,视差大小是160,所以要3个64，64-128
	//int y3=blockIdx.y*blockDim.y+threadIdx.y+128-disp_max;//128-192
	//output[threadIdx.x][threadIdx.y]=0;
	if((y>disp_max+(kernel_size-1)/2+box_kernel)&&(y<cols-disp_max-(kernel_size-1)/2-box_kernel)&&(x>(kernel_size-1)/2+box_kernel+filterweight_kernel)&&(x<rows-(kernel_size-1)/2-box_kernel-filterweight_kernel))//y>disp_max，表示了左边disp_max个点不要了
	{
		//configure left shared memory
		sh_pic[threadIdx.x][threadIdx.y]=a[x-filterweight_kernel][y-filterweight_kernel];//因为共享内存上下左右各大了一行，而坐标原点在左上角，所以需要全部-1
		sh_disp[threadIdx.x][threadIdx.y]=b[x-filterweight_kernel][y-filterweight_kernel];
		if(threadIdx.x<(2*filterweight_kernel))//内存重复覆盖的问题？这里这个threadIdx.x是不是一定小于2,因为block_size_x是2，所以判断语句是不是不需要了？
		{
			sh_pic[threadIdx.x+block_size_x][threadIdx.y]=a[x+block_size_x-filterweight_kernel][y-filterweight_kernel];//相当于存了4行的数据，上下各拓宽了一行
			sh_disp[threadIdx.x+block_size_x][threadIdx.y]=b[x+block_size_x-filterweight_kernel][y-filterweight_kernel];
		}
		if(threadIdx.y<(2*filterweight_kernel))//这个是为了扩充右边界，因为上下左右各拓宽了一行
		{
			sh_pic[threadIdx.x][threadIdx.y+block_size_y]=a[x-filterweight_kernel][y+block_size_y-filterweight_kernel];
			sh_disp[threadIdx.x][threadIdx.y+block_size_y]=b[x-filterweight_kernel][y+block_size_y-filterweight_kernel];
		}
		if((threadIdx.x<2*filterweight_kernel)&&(threadIdx.y<2*filterweight_kernel))//这个是为了更新边框右下角的数据
		{
			sh_pic[threadIdx.x+block_size_x][threadIdx.y+block_size_y]=a[x+block_size_x-filterweight_kernel][y+block_size_y-filterweight_kernel];
			sh_disp[threadIdx.x+block_size_x][threadIdx.y+block_size_y]=b[x+block_size_x-filterweight_kernel][y+block_size_y-filterweight_kernel];
		}
		__syncthreads();

		float temp_pic_c[2*filterweight_kernel+1][2*filterweight_kernel+1];
		float temp_pic_a[2*filterweight_kernel+1][2*filterweight_kernel+1];//color+distance=all,temp_pic_d
		float temp_disp[2*filterweight_kernel+1][2*filterweight_kernel+1];

		int model1=255;
		int model2=255*2^8;
		int temp1_l=0;
		int temp1_r=0;
		int temp2_l=0;
		int temp2_r=0;
		int temp3_l=0;
		int temp3_r=0;


		temp1_r=(sh_pic[threadIdx.x+filterweight_kernel][threadIdx.y+filterweight_kernel])&model1;//这里1r存的是中心点的低8位值
		temp2_r=(sh_pic[threadIdx.x+filterweight_kernel][threadIdx.y+filterweight_kernel])&model2;//这里2r存的是中心点的中8位值
		temp2_r=temp2_r>>8;
		temp3_r=(sh_pic[threadIdx.x+filterweight_kernel][threadIdx.y+filterweight_kernel])>>16;//这里3r存的是中心点的高8位值
		for(int i=0;i<(2*filterweight_kernel+1);i++)
		{
			for(int j=0;j<(2*filterweight_kernel+1);j++)//关于颜色的地方需要改改！！RGB
			{
				temp1_l=(sh_pic[threadIdx.x+i][threadIdx.y+j])&model1;
				temp2_l=(sh_pic[threadIdx.x+i][threadIdx.y+j])&model2;
				temp2_l=temp2_l>>8;
				temp3_l=(sh_pic[threadIdx.x+i][threadIdx.y+j])>>16;
				temp_pic_c[i][j]=sqrt((float)((temp1_l-temp1_r)*(temp1_l-temp1_r)+(temp2_l-temp2_r)*(temp2_l-temp2_r)+(temp3_l-temp3_r)*(temp3_l-temp3_r)));
			}
		}
		for(int i=0;i<(2*filterweight_kernel+1);i++)
		{
			for(int j=0;j<(2*filterweight_kernel+1);j++)
			{
				temp_pic_a[i][j]=exp(-1*(temp_pic_c[i][j]/sigma_c)*(temp_pic_c[i][j]/sigma_c))*exp(-1*(temp_d[i][j]/sigma_p)*(temp_d[i][j]/sigma_p));//sigma_c和sigma_p需要定义
			}
		}
//		for(int i=0;i<(2*filterweight_kernel+1);i++)
//		{
//			for(int j=0;j<(2*filterweight_kernel+1);j++)
//			{
//				temp_disp[i][j]=temp_pic_a[i][j]*(float)sh_disp[threadIdx.x+i][threadIdx.y+j];//sigma_c和sigma_p需要定义
//			}
//		}
		//Middle filter
		float temp_disp_middle[(2*filterweight_kernel+1)*(2*filterweight_kernel+1)];
		float temp_pic_middle[(2*filterweight_kernel+1)*(2*filterweight_kernel+1)];
		for(int i=0;i<(2*filterweight_kernel+1);i++)
		{
			for(int j=0;j<(2*filterweight_kernel+1);j++)
			{
				temp_disp_middle[i*(2*filterweight_kernel+1)+j]=(float)sh_disp[threadIdx.x+i][threadIdx.y+j];
				temp_pic_middle[i*(2*filterweight_kernel+1)+j]=temp_pic_a[i][j];
			}
		}

		for (int j=0; j<(2*filterweight_kernel+1); ++j)
		    {
		        int min_middle=j;
		        for (int l=j+1; l<(2*filterweight_kernel+1)*(2*filterweight_kernel+1)-1; ++l)
		            if (temp_disp_middle[l] < temp_disp_middle[min_middle])
		            	min_middle=l;
		        //视差排序同时权值排序
		        const float bbtemp=temp_disp_middle[j];
		        temp_disp_middle[j]=temp_disp_middle[min_middle];
		        temp_disp_middle[min_middle]=bbtemp;

		        const float bbbtemp=temp_pic_middle[j];
		        temp_pic_middle[j]=temp_pic_middle[min_middle];
		        temp_pic_middle[min_middle]=bbbtemp;
		    }
		float SUM=0;
		for (int i=0;i<(2*filterweight_kernel+1)*(2*filterweight_kernel+1);i++)
		{
			SUM= SUM + temp_pic_middle[i];
		}
		SUM=SUM*MIDDLE_kernel;
		float Mid_SUM=0;
		for (int i=0;i<(2*filterweight_kernel+1);i++)
		{
			for (int j=0;j<(2*filterweight_kernel+1);j++)
			{
				Mid_SUM= Mid_SUM + temp_pic_middle[j+(2*filterweight_kernel+1)*i];
				if(Mid_SUM>=SUM)
				{
					output[x][y]= temp_disp_middle[j+(2*filterweight_kernel+1)*i];
					i=2*filterweight_kernel+1;
					break;
				}
			}
		}

//		for(int i=0;i<(2*filterweight_kernel+1); ++i)
//		{
//			for(int j=0;j<(2*filterweight_kernel+1);j++)
//			{
//				if(temp_disp[i][j]==temp_middle[filterweight_kernel])
//				{
//					output[x][y]= sh_disp[i+threadIdx.x][j+threadIdx.y];
//					i=2*filterweight_kernel+1;
//					break;
//				}
//			}
//		}
	}
}


__global__ void filter_weight_h(uint (*a)[cols],uchar (*b)[cols],uchar (*output)[cols],float (*temp_d)[2*filterweight_kernel_h+1])//filterweight_kernel_h
{
	//这里只要2*filterweight_kernel_h<block_size_x和block_size_y就行了
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	output[x][y]=0;
	//test0=10;
	//因为用的是3X3的区域内计算，所以上下左右方向各扩充一格

	__shared__ uint sh_pic[block_size_x+filterweight_kernel_h*2][block_size_y+filterweight_kernel_h*2];//共享内存为什么要大点，共享内存只能在方格内部起作用
	__shared__ uint sh_disp[block_size_x+filterweight_kernel_h*2][block_size_y+filterweight_kernel_h*2];
	//这些int都相当于寄存器
	//int y1=blockIdx.y*blockDim.y+threadIdx.y-disp_max;//这个定义中64,128还有就是disp_max作用？0-64
	//int y2=blockIdx.y*blockDim.y+threadIdx.y+64-disp_max;//block_size_y的大小是64,视差大小是160,所以要3个64，64-128
	//int y3=blockIdx.y*blockDim.y+threadIdx.y+128-disp_max;//128-192
	//output[threadIdx.x][threadIdx.y]=0;
	if((y>disp_max+(kernel_size-1)/2+box_kernel)&&(y<cols-disp_max-(kernel_size-1)/2-box_kernel)&&(x>(kernel_size-1)/2+box_kernel+filterweight_kernel_h)&&(x<rows-(kernel_size-1)/2-box_kernel-filterweight_kernel_h))//y>disp_max，表示了左边disp_max个点不要了
	{
		//configure left shared memory
		sh_pic[threadIdx.x][threadIdx.y]=a[x-filterweight_kernel_h][y-filterweight_kernel_h];//因为共享内存上下左右各大了一行，而坐标原点在左上角，所以需要全部-1
		sh_disp[threadIdx.x][threadIdx.y]=b[x-filterweight_kernel_h][y-filterweight_kernel_h];
		if(threadIdx.x<(2*filterweight_kernel_h))//内存重复覆盖的问题？这里这个threadIdx.x是不是一定小于2,因为block_size_x是2，所以判断语句是不是不需要了？
		{
			sh_pic[threadIdx.x+block_size_x][threadIdx.y]=a[x+block_size_x-filterweight_kernel_h][y-filterweight_kernel_h];//相当于存了4行的数据，上下各拓宽了一行
			sh_disp[threadIdx.x+block_size_x][threadIdx.y]=b[x+block_size_x-filterweight_kernel_h][y-filterweight_kernel_h];
		}
		if(threadIdx.y<(2*filterweight_kernel_h))//这个是为了扩充右边界，因为上下左右各拓宽了一行
		{
			sh_pic[threadIdx.x][threadIdx.y+block_size_y]=a[x-filterweight_kernel_h][y+block_size_y-filterweight_kernel_h];
			sh_disp[threadIdx.x][threadIdx.y+block_size_y]=b[x-filterweight_kernel_h][y+block_size_y-filterweight_kernel_h];
		}
		if((threadIdx.x<2*filterweight_kernel_h)&&(threadIdx.y<2*filterweight_kernel_h))//这个是为了更新边框右下角的数据
		{
			sh_pic[threadIdx.x+block_size_x][threadIdx.y+block_size_y]=a[x+block_size_x-filterweight_kernel_h][y+block_size_y-filterweight_kernel_h];
			sh_disp[threadIdx.x+block_size_x][threadIdx.y+block_size_y]=b[x+block_size_x-filterweight_kernel_h][y+block_size_y-filterweight_kernel_h];
		}
		__syncthreads();

		float temp_pic_c[2*filterweight_kernel_h+1][2*filterweight_kernel_h+1];
		float temp_pic_a[2*filterweight_kernel_h+1][2*filterweight_kernel_h+1];//color+distance=all,temp_pic_d
		float temp_disp[2*filterweight_kernel_h+1][2*filterweight_kernel_h+1];

		int model1=255;
		int model2=255*2^8;
		int temp1_l=0;
		int temp1_r=0;
		int temp2_l=0;
		int temp2_r=0;
		int temp3_l=0;
		int temp3_r=0;


		temp1_r=(sh_pic[threadIdx.x+filterweight_kernel_h][threadIdx.y+filterweight_kernel_h])&model1;//这里1r存的是中心点的低8位值
		temp2_r=(sh_pic[threadIdx.x+filterweight_kernel_h][threadIdx.y+filterweight_kernel_h])&model2;//这里2r存的是中心点的中8位值
		temp2_r=temp2_r>>8;
		temp3_r=(sh_pic[threadIdx.x+filterweight_kernel_h][threadIdx.y+filterweight_kernel_h])>>16;//这里3r存的是中心点的高8位值
		for(int i=0;i<(2*filterweight_kernel_h+1);i++)
		{
			for(int j=0;j<(2*filterweight_kernel_h+1);j++)//关于颜色的地方需要改改！！RGB
			{
				temp1_l=(sh_pic[threadIdx.x+i][threadIdx.y+j])&model1;
				temp2_l=(sh_pic[threadIdx.x+i][threadIdx.y+j])&model2;
				temp2_l=temp2_l>>8;
				temp3_l=(sh_pic[threadIdx.x+i][threadIdx.y+j])>>16;
				temp_pic_c[i][j]=sqrt((float)((temp1_l-temp1_r)*(temp1_l-temp1_r)+(temp2_l-temp2_r)*(temp2_l-temp2_r)+(temp3_l-temp3_r)*(temp3_l-temp3_r)));
			}
		}
		for(int i=0;i<(2*filterweight_kernel_h+1);i++)
		{
			for(int j=0;j<(2*filterweight_kernel_h+1);j++)
			{
				temp_pic_a[i][j]=exp(-1*(temp_pic_c[i][j]/sigma_c)*(temp_pic_c[i][j]/sigma_c))*exp(-1*(temp_d[i][j]/sigma_p)*(temp_d[i][j]/sigma_p));//sigma_c和sigma_p需要定义
			}
		}
//		for(int i=0;i<(2*filterweight_kernel_h+1);i++)
//		{
//			for(int j=0;j<(2*filterweight_kernel_h+1);j++)
//			{
//				temp_disp[i][j]=temp_pic_a[i][j]*(float)sh_disp[threadIdx.x+i][threadIdx.y+j];//sigma_c和sigma_p需要定义
//			}
//		}
		//Middle filter
		float temp_disp_middle[(2*filterweight_kernel_h+1)*(2*filterweight_kernel_h+1)];
		float temp_pic_middle[(2*filterweight_kernel_h+1)*(2*filterweight_kernel_h+1)];
		for(int i=0;i<(2*filterweight_kernel_h+1);i++)
		{
			for(int j=0;j<(2*filterweight_kernel_h+1);j++)
			{
				temp_disp_middle[i*(2*filterweight_kernel_h+1)+j]=(float)sh_disp[threadIdx.x+i][threadIdx.y+j];
				temp_pic_middle[i*(2*filterweight_kernel_h+1)+j]=temp_pic_a[i][j];
			}
		}

		for (int j=0; j<(2*filterweight_kernel_h+1); ++j)
		    {
		        int min_middle=j;
		        for (int l=j+1; l<(2*filterweight_kernel_h+1)*(2*filterweight_kernel_h+1)-1; ++l)
		            if (temp_disp_middle[l] < temp_disp_middle[min_middle])
		            	min_middle=l;
		        //视差排序同时权值排序
		        const float bbtemp=temp_disp_middle[j];
		        temp_disp_middle[j]=temp_disp_middle[min_middle];
		        temp_disp_middle[min_middle]=bbtemp;

		        const float bbbtemp=temp_pic_middle[j];
		        temp_pic_middle[j]=temp_pic_middle[min_middle];
		        temp_pic_middle[min_middle]=bbbtemp;
		    }
		float SUM=0;
		for (int i=0;i<(2*filterweight_kernel_h+1)*(2*filterweight_kernel_h+1);i++)
		{
			SUM= SUM + temp_pic_middle[i];
		}
		SUM=SUM*MIDDLE_kernel_h;
		float Mid_SUM=0;
		for (int i=0;i<(2*filterweight_kernel_h+1);i++)
		{
			for (int j=0;j<(2*filterweight_kernel_h+1);j++)
			{
				Mid_SUM= Mid_SUM + temp_pic_middle[j+(2*filterweight_kernel_h+1)*i];
				if(Mid_SUM>=SUM)
				{
					output[x][y]= temp_disp_middle[j+(2*filterweight_kernel_h+1)*i];
					i=2*filterweight_kernel_h+1;
					break;
				}
			}
		}

//		for(int i=0;i<(2*filterweight_kernel_h+1); ++i)
//		{
//			for(int j=0;j<(2*filterweight_kernel_h+1);j++)
//			{
//				if(temp_disp[i][j]==temp_middle[filterweight_kernel_h])
//				{
//					output[x][y]= sh_disp[i+threadIdx.x][j+threadIdx.y];
//					i=2*filterweight_kernel_h+1;
//					break;
//				}
//			}
//		}
	}
}





int main()
{
	//cudaSetDevice(0);
	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//deviceProp.unifiedAddressing=0;

	dim3 threads(block_size_x,block_size_y);
	dim3 blocks(iDivUp(rows,block_size_x),iDivUp(cols,block_size_y));
	float temp_pic_d[2*filterweight_kernel+1][2*filterweight_kernel+1];
	float temp_pic_d_h[2*filterweight_kernel_h+1][2*filterweight_kernel_h+1];

	uint (*cpu_p1)[cols];
	uint (*cpu_p2)[cols];
	uint (*gpu_p1)[cols];
	uint (*gpu_p1_flip)[cols];
	uint (*gpu_p2)[cols];
	uint (*gpu_p2_flip)[cols];
	uchar (*gpu_p3)[cols];
	uchar (*gpu_p4)[cols];
	uchar (*gpu_p5)[cols];
	uchar (*gpu_p6)[cols];
	uchar (*dr)[cols];
	uchar (*dd)[cols];
	uchar (*dl)[cols];
	uchar (*du)[cols];
	uchar (*dlr)[cols];
	uchar (*dud)[cols];
	uchar (*gpu_p7)[cols];
	bool (*gpu_LR_error)[cols];
	bool (*gpu_LR_error_R)[cols];
	float (*temp_pic_disp)[2*filterweight_kernel+1];
	float (*temp_pic_disp_h)[2*filterweight_kernel_h+1];

	uchar (*gpu_p8)[cols];
	uchar (*gpu_p9)[cols];
	uchar (*gpu_p10)[cols];
	uchar (*gpu_p11)[cols];
	uchar (*gpu_p12)[cols];
	uchar (*gpu_p13)[cols];
	uchar (*gpu_p14)[cols];
	uchar (*gpu_p15)[cols];
	uchar (*gpu_p16)[cols];
	uchar (*gpu_p17)[cols];
	uchar (*gpu_p18)[cols];
	uchar (*gpu_p19)[cols];

	Mat im1,im2,im3,im4;
	im3.create(rows,cols,CV_8UC1);
	im4.create(rows,cols,CV_8UC1);
	im1=imread("im0.png");
	im2=imread("im1.png");

//	imshow("左图",im1);

	//锁页内存
	cudaHostAlloc( (void**)&cpu_p1,rows*cols*sizeof(uint),cudaHostAllocDefault);
	cudaHostAlloc( (void**)&cpu_p2,rows*cols*sizeof(uint),cudaHostAllocDefault);


	for(int x=0;x<rows;x++)
	{
		for(int y=0;y<cols;y++)
		{
			cpu_p1[x][y]=im1.at<Vec3b>(x,y)[0]+(im1.at<Vec3b>(x,y)[1]<<8)+(im1.at<Vec3b>(x,y)[2]<<16);//这个左移神马意思？
			cpu_p2[x][y]=im2.at<Vec3b>(x,y)[0]+(im2.at<Vec3b>(x,y)[1]<<8)+(im2.at<Vec3b>(x,y)[2]<<16);//读进来是char型，cpu_p1是int型
		}
	}


	for(int i=0;i<(2*filterweight_kernel+1);i++)
	{
		for(int j=0;j<(2*filterweight_kernel+1);j++)
		{
			temp_pic_d[i][j]=(i-filterweight_kernel)*(i-filterweight_kernel)+(j-filterweight_kernel)*(j-filterweight_kernel);
		}
	}
	for(int i=0;i<(2*filterweight_kernel_h+1);i++)
	{
		for(int j=0;j<(2*filterweight_kernel_h+1);j++)
		{
			temp_pic_d_h[i][j]=(i-filterweight_kernel_h)*(i-filterweight_kernel_h)+(j-filterweight_kernel_h)*(j-filterweight_kernel_h);
		}
	}



	//自动补位
	size_t pitch;
	cudaMallocPitch((void **)&gpu_p1,&pitch,cols*sizeof(uint),rows);//这个pitch自动补位存放的是补位值，如果覆盖了以后就无法访问了
	cudaMallocPitch((void **)&gpu_p2,&pitch,cols*sizeof(uint),rows);//我理解这么做是为了运行快
	cudaMallocPitch((void **)&gpu_p1_flip,&pitch,cols*sizeof(uint),rows);//我理解这么做是为了运行快
	cudaMallocPitch((void **)&gpu_p2_flip,&pitch,cols*sizeof(uint),rows);//我理解这么做是为了运行快
	cudaMallocPitch((void **)&gpu_p3,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p4,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p5,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p6,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&temp,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&dr,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&dd,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&dl,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&du,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&dlr,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&dud,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p7,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_LR_error,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_LR_error_R,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p8,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p9,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p10,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p11,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p12,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p13,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&temp_pic_disp,&pitch,(2*filterweight_kernel+1)*sizeof(float),(2*filterweight_kernel+1));
	cudaMallocPitch((void **)&temp_pic_disp_h,&pitch,(2*filterweight_kernel_h+1)*sizeof(float),(2*filterweight_kernel_h+1));
	cudaMallocPitch((void **)&gpu_p14,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p15,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p16,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p17,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p18,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch((void **)&gpu_p19,&pitch,cols*sizeof(uchar),rows);

	cudaMemcpyAsync(gpu_p1,cpu_p1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);//数据传输阻塞和非阻塞，同步和异步？
	cudaMemcpyAsync(gpu_p2,cpu_p2,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	cudaMemcpyAsync(temp_pic_disp,temp_pic_d,(2*filterweight_kernel+1)*(2*filterweight_kernel+1)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpyAsync(temp_pic_disp_h,temp_pic_d_h,(2*filterweight_kernel_h+1)*(2*filterweight_kernel_h+1)*sizeof(float),cudaMemcpyHostToDevice);
//	cudaEvent_t start,stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start,0);
//
//	cudaEventRecord(stop,0);
//	cudaEventSynchronize(stop);
//	float time;
//	cudaEventElapsedTime(&time,start,stop);
//	printf("Time is %fms\n",time);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	flip_kernel<<<blocks,threads>>>(gpu_p1,gpu_p1_flip);
	flip_kernel<<<blocks,threads>>>(gpu_p2,gpu_p2_flip);
	stereo_kernel<<<blocks,threads>>>(gpu_p1,gpu_p2,gpu_p3);
	stereo_kernel<<<blocks,threads>>>(gpu_p2_flip,gpu_p1_flip,gpu_p5);
	flip_kernel_char<<<blocks,threads>>>(gpu_p5,gpu_p6);

	Match_error<<<blocks,threads>>>(gpu_p3,gpu_p6,gpu_LR_error);
	near_match<<<blocks,threads>>>(	gpu_p3,gpu_p7,gpu_LR_error,dr,
									dl,du,dd,
									dlr,dud);
//
	Match_error<<<blocks,threads>>>(gpu_p6,gpu_p3,gpu_LR_error_R);
	near_match<<<blocks,threads>>>(	gpu_p6,gpu_p15,gpu_LR_error_R,dr,
									dl,du,dd,
									dlr,dud);


//	middle_filter<<<blocks,threads>>>(gpu_p7,gpu_p10,Middle_Radius);
	filter_weight<<<blocks,threads>>>(gpu_p1,gpu_p3,gpu_p13,temp_pic_disp);
	filter_weight<<<blocks,threads>>>(gpu_p2,gpu_p15,gpu_p14,temp_pic_disp);

	filter_weight_h<<<blocks,threads>>>(gpu_p1,gpu_p13,gpu_p17,temp_pic_disp_h);
	filter_weight_h<<<blocks,threads>>>(gpu_p2,gpu_p14,gpu_p16,temp_pic_disp_h);
	filter_weight_h<<<blocks,threads>>>(gpu_p2,gpu_p16,gpu_p18,temp_pic_disp_h);
	filter_weight_h<<<blocks,threads>>>(gpu_p2,gpu_p18,gpu_p19,temp_pic_disp_h);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time,start,stop);
	printf("Time is %fms\n",time);
//	box_filter(gpu_p7,gpu_p4,0,blocks,threads);
//	pic_show<<<blocks,threads>>>(gpu_p13,gpu_p11);
//	pic_show<<<blocks,threads>>>(gpu_p3,gpu_p12);





	cudaMemcpy(im3.data,gpu_p17,rows*cols*sizeof(uchar),cudaMemcpyDeviceToHost);
	cudaMemcpy(im4.data,gpu_p19,rows*cols*sizeof(uchar),cudaMemcpyDeviceToHost);
	//imshow("视差图l",im3);
	//imshow("视差图r",im4);
	imwrite("disp_l.bmp",im3);
	imwrite("disp_r.bmp",im4);
	waitKey(0);
	return 0;
}
