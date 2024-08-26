#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=8:00:00
#SBATCH --job-name=save_cats
#SBATCH -p gpu
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err

module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

cd /mnt/home/spandey/ceph/CHARM/charm/
srun python predict_save_halo_cats.py 0
srun python predict_save_halo_cats.py 1
srun python predict_save_halo_cats.py 2
srun python predict_save_halo_cats.py 3
srun python predict_save_halo_cats.py 4
srun python predict_save_halo_cats.py 5
srun python predict_save_halo_cats.py 6
srun python predict_save_halo_cats.py 7
srun python predict_save_halo_cats.py 8
srun python predict_save_halo_cats.py 9
srun python predict_save_halo_cats.py 10
srun python predict_save_halo_cats.py 11
srun python predict_save_halo_cats.py 12
srun python predict_save_halo_cats.py 13
srun python predict_save_halo_cats.py 14
srun python predict_save_halo_cats.py 15
srun python predict_save_halo_cats.py 16
srun python predict_save_halo_cats.py 17
srun python predict_save_halo_cats.py 18
srun python predict_save_halo_cats.py 19
srun python predict_save_halo_cats.py 20
srun python predict_save_halo_cats.py 21
srun python predict_save_halo_cats.py 22
srun python predict_save_halo_cats.py 23
srun python predict_save_halo_cats.py 24
srun python predict_save_halo_cats.py 25
srun python predict_save_halo_cats.py 26
srun python predict_save_halo_cats.py 27
srun python predict_save_halo_cats.py 28
srun python predict_save_halo_cats.py 29
srun python predict_save_halo_cats.py 30
srun python predict_save_halo_cats.py 31
srun python predict_save_halo_cats.py 32
srun python predict_save_halo_cats.py 33
srun python predict_save_halo_cats.py 34
srun python predict_save_halo_cats.py 35
srun python predict_save_halo_cats.py 36
srun python predict_save_halo_cats.py 37
srun python predict_save_halo_cats.py 38
srun python predict_save_halo_cats.py 39
srun python predict_save_halo_cats.py 40
srun python predict_save_halo_cats.py 41
srun python predict_save_halo_cats.py 42
srun python predict_save_halo_cats.py 43
srun python predict_save_halo_cats.py 44
srun python predict_save_halo_cats.py 45
srun python predict_save_halo_cats.py 46
srun python predict_save_halo_cats.py 47
srun python predict_save_halo_cats.py 48
srun python predict_save_halo_cats.py 49
srun python predict_save_halo_cats.py 50
srun python predict_save_halo_cats.py 51
srun python predict_save_halo_cats.py 52
srun python predict_save_halo_cats.py 53
srun python predict_save_halo_cats.py 54
srun python predict_save_halo_cats.py 55
srun python predict_save_halo_cats.py 56
srun python predict_save_halo_cats.py 57
srun python predict_save_halo_cats.py 58
srun python predict_save_halo_cats.py 59
srun python predict_save_halo_cats.py 60
srun python predict_save_halo_cats.py 61
srun python predict_save_halo_cats.py 62
srun python predict_save_halo_cats.py 63
srun python predict_save_halo_cats.py 64
srun python predict_save_halo_cats.py 65
srun python predict_save_halo_cats.py 66
srun python predict_save_halo_cats.py 67
srun python predict_save_halo_cats.py 68
srun python predict_save_halo_cats.py 69
srun python predict_save_halo_cats.py 70
srun python predict_save_halo_cats.py 71
srun python predict_save_halo_cats.py 72
srun python predict_save_halo_cats.py 73
srun python predict_save_halo_cats.py 74
srun python predict_save_halo_cats.py 75
srun python predict_save_halo_cats.py 76
srun python predict_save_halo_cats.py 77
srun python predict_save_halo_cats.py 78
srun python predict_save_halo_cats.py 79
srun python predict_save_halo_cats.py 80
srun python predict_save_halo_cats.py 81
srun python predict_save_halo_cats.py 82
srun python predict_save_halo_cats.py 83
srun python predict_save_halo_cats.py 84
srun python predict_save_halo_cats.py 85
srun python predict_save_halo_cats.py 86
srun python predict_save_halo_cats.py 87
srun python predict_save_halo_cats.py 88
srun python predict_save_halo_cats.py 89
srun python predict_save_halo_cats.py 90
srun python predict_save_halo_cats.py 91
srun python predict_save_halo_cats.py 92
srun python predict_save_halo_cats.py 93
srun python predict_save_halo_cats.py 94
srun python predict_save_halo_cats.py 95
srun python predict_save_halo_cats.py 96
srun python predict_save_halo_cats.py 97
srun python predict_save_halo_cats.py 98
srun python predict_save_halo_cats.py 99
srun python predict_save_halo_cats.py 100
srun python predict_save_halo_cats.py 101
srun python predict_save_halo_cats.py 102
srun python predict_save_halo_cats.py 103
srun python predict_save_halo_cats.py 104
srun python predict_save_halo_cats.py 105
srun python predict_save_halo_cats.py 106
srun python predict_save_halo_cats.py 107
srun python predict_save_halo_cats.py 108
srun python predict_save_halo_cats.py 109
srun python predict_save_halo_cats.py 110
srun python predict_save_halo_cats.py 111
srun python predict_save_halo_cats.py 112
srun python predict_save_halo_cats.py 113
srun python predict_save_halo_cats.py 114
srun python predict_save_halo_cats.py 115
srun python predict_save_halo_cats.py 116
srun python predict_save_halo_cats.py 117
srun python predict_save_halo_cats.py 118
srun python predict_save_halo_cats.py 119
srun python predict_save_halo_cats.py 120
srun python predict_save_halo_cats.py 121
srun python predict_save_halo_cats.py 122
srun python predict_save_halo_cats.py 123
srun python predict_save_halo_cats.py 124
srun python predict_save_halo_cats.py 125
srun python predict_save_halo_cats.py 126
srun python predict_save_halo_cats.py 127
srun python predict_save_halo_cats.py 128
srun python predict_save_halo_cats.py 129
srun python predict_save_halo_cats.py 130
srun python predict_save_halo_cats.py 131
srun python predict_save_halo_cats.py 132
srun python predict_save_halo_cats.py 133
srun python predict_save_halo_cats.py 134
srun python predict_save_halo_cats.py 135
srun python predict_save_halo_cats.py 136
srun python predict_save_halo_cats.py 137
srun python predict_save_halo_cats.py 138
srun python predict_save_halo_cats.py 139
srun python predict_save_halo_cats.py 140
srun python predict_save_halo_cats.py 141
srun python predict_save_halo_cats.py 142
srun python predict_save_halo_cats.py 143
srun python predict_save_halo_cats.py 144
srun python predict_save_halo_cats.py 145
srun python predict_save_halo_cats.py 146
srun python predict_save_halo_cats.py 147
srun python predict_save_halo_cats.py 148
srun python predict_save_halo_cats.py 149
srun python predict_save_halo_cats.py 150
srun python predict_save_halo_cats.py 151
srun python predict_save_halo_cats.py 152
srun python predict_save_halo_cats.py 153
srun python predict_save_halo_cats.py 154
srun python predict_save_halo_cats.py 155
srun python predict_save_halo_cats.py 156
srun python predict_save_halo_cats.py 157
srun python predict_save_halo_cats.py 158
srun python predict_save_halo_cats.py 159
srun python predict_save_halo_cats.py 160
srun python predict_save_halo_cats.py 161
srun python predict_save_halo_cats.py 162
srun python predict_save_halo_cats.py 163
srun python predict_save_halo_cats.py 164
srun python predict_save_halo_cats.py 165
srun python predict_save_halo_cats.py 166
srun python predict_save_halo_cats.py 167
srun python predict_save_halo_cats.py 168
srun python predict_save_halo_cats.py 169
srun python predict_save_halo_cats.py 170
srun python predict_save_halo_cats.py 171
srun python predict_save_halo_cats.py 172
srun python predict_save_halo_cats.py 173
srun python predict_save_halo_cats.py 174
srun python predict_save_halo_cats.py 175
srun python predict_save_halo_cats.py 176
srun python predict_save_halo_cats.py 177
srun python predict_save_halo_cats.py 178
srun python predict_save_halo_cats.py 179
srun python predict_save_halo_cats.py 180
srun python predict_save_halo_cats.py 181
srun python predict_save_halo_cats.py 182
srun python predict_save_halo_cats.py 183
srun python predict_save_halo_cats.py 184
srun python predict_save_halo_cats.py 185
srun python predict_save_halo_cats.py 186
srun python predict_save_halo_cats.py 187
srun python predict_save_halo_cats.py 188
srun python predict_save_halo_cats.py 189
srun python predict_save_halo_cats.py 190
srun python predict_save_halo_cats.py 191
srun python predict_save_halo_cats.py 192
srun python predict_save_halo_cats.py 193
srun python predict_save_halo_cats.py 194
srun python predict_save_halo_cats.py 195
srun python predict_save_halo_cats.py 196
srun python predict_save_halo_cats.py 197
srun python predict_save_halo_cats.py 198
srun python predict_save_halo_cats.py 199
srun python predict_save_halo_cats.py 200
srun python predict_save_halo_cats.py 201
srun python predict_save_halo_cats.py 202
srun python predict_save_halo_cats.py 203
srun python predict_save_halo_cats.py 204
srun python predict_save_halo_cats.py 205
srun python predict_save_halo_cats.py 206
srun python predict_save_halo_cats.py 207
srun python predict_save_halo_cats.py 208
srun python predict_save_halo_cats.py 209
srun python predict_save_halo_cats.py 210
srun python predict_save_halo_cats.py 211
srun python predict_save_halo_cats.py 212
srun python predict_save_halo_cats.py 213
srun python predict_save_halo_cats.py 214
srun python predict_save_halo_cats.py 215
srun python predict_save_halo_cats.py 216
srun python predict_save_halo_cats.py 217
srun python predict_save_halo_cats.py 218
srun python predict_save_halo_cats.py 219
srun python predict_save_halo_cats.py 220
srun python predict_save_halo_cats.py 221
srun python predict_save_halo_cats.py 222
srun python predict_save_halo_cats.py 223
srun python predict_save_halo_cats.py 224
srun python predict_save_halo_cats.py 225
srun python predict_save_halo_cats.py 226
srun python predict_save_halo_cats.py 227
srun python predict_save_halo_cats.py 228
srun python predict_save_halo_cats.py 229
srun python predict_save_halo_cats.py 230
srun python predict_save_halo_cats.py 231
srun python predict_save_halo_cats.py 232
srun python predict_save_halo_cats.py 233
srun python predict_save_halo_cats.py 234
srun python predict_save_halo_cats.py 235
srun python predict_save_halo_cats.py 236
srun python predict_save_halo_cats.py 237
srun python predict_save_halo_cats.py 238
srun python predict_save_halo_cats.py 239
srun python predict_save_halo_cats.py 240
srun python predict_save_halo_cats.py 241
srun python predict_save_halo_cats.py 242
srun python predict_save_halo_cats.py 243
srun python predict_save_halo_cats.py 244
srun python predict_save_halo_cats.py 245
srun python predict_save_halo_cats.py 246
srun python predict_save_halo_cats.py 247
srun python predict_save_halo_cats.py 248
srun python predict_save_halo_cats.py 249
srun python predict_save_halo_cats.py 250
srun python predict_save_halo_cats.py 251
srun python predict_save_halo_cats.py 252
srun python predict_save_halo_cats.py 253
srun python predict_save_halo_cats.py 254
srun python predict_save_halo_cats.py 255
srun python predict_save_halo_cats.py 256
srun python predict_save_halo_cats.py 257
srun python predict_save_halo_cats.py 258
srun python predict_save_halo_cats.py 259
srun python predict_save_halo_cats.py 260
srun python predict_save_halo_cats.py 261
srun python predict_save_halo_cats.py 262
srun python predict_save_halo_cats.py 263
srun python predict_save_halo_cats.py 264
srun python predict_save_halo_cats.py 265
srun python predict_save_halo_cats.py 266
srun python predict_save_halo_cats.py 267
srun python predict_save_halo_cats.py 268
srun python predict_save_halo_cats.py 269
srun python predict_save_halo_cats.py 270
srun python predict_save_halo_cats.py 271
srun python predict_save_halo_cats.py 272
srun python predict_save_halo_cats.py 273
srun python predict_save_halo_cats.py 274
srun python predict_save_halo_cats.py 275
srun python predict_save_halo_cats.py 276
srun python predict_save_halo_cats.py 277
srun python predict_save_halo_cats.py 278
srun python predict_save_halo_cats.py 279
srun python predict_save_halo_cats.py 280
srun python predict_save_halo_cats.py 281
srun python predict_save_halo_cats.py 282
srun python predict_save_halo_cats.py 283
srun python predict_save_halo_cats.py 284
srun python predict_save_halo_cats.py 285
srun python predict_save_halo_cats.py 286
srun python predict_save_halo_cats.py 287
srun python predict_save_halo_cats.py 288
srun python predict_save_halo_cats.py 289
srun python predict_save_halo_cats.py 290
srun python predict_save_halo_cats.py 291
srun python predict_save_halo_cats.py 292
srun python predict_save_halo_cats.py 293
srun python predict_save_halo_cats.py 294
srun python predict_save_halo_cats.py 295
srun python predict_save_halo_cats.py 296
srun python predict_save_halo_cats.py 297
srun python predict_save_halo_cats.py 298
srun python predict_save_halo_cats.py 299
srun python predict_save_halo_cats.py 300
srun python predict_save_halo_cats.py 301
srun python predict_save_halo_cats.py 302
srun python predict_save_halo_cats.py 303
srun python predict_save_halo_cats.py 304
srun python predict_save_halo_cats.py 305
srun python predict_save_halo_cats.py 306
srun python predict_save_halo_cats.py 307
srun python predict_save_halo_cats.py 308
srun python predict_save_halo_cats.py 309
srun python predict_save_halo_cats.py 310
srun python predict_save_halo_cats.py 311
srun python predict_save_halo_cats.py 312
srun python predict_save_halo_cats.py 313
srun python predict_save_halo_cats.py 314
srun python predict_save_halo_cats.py 315
srun python predict_save_halo_cats.py 316
srun python predict_save_halo_cats.py 317
srun python predict_save_halo_cats.py 318
srun python predict_save_halo_cats.py 319
srun python predict_save_halo_cats.py 320
srun python predict_save_halo_cats.py 321
srun python predict_save_halo_cats.py 322
srun python predict_save_halo_cats.py 323
srun python predict_save_halo_cats.py 324
srun python predict_save_halo_cats.py 325
srun python predict_save_halo_cats.py 326
srun python predict_save_halo_cats.py 327
srun python predict_save_halo_cats.py 328
srun python predict_save_halo_cats.py 329
srun python predict_save_halo_cats.py 330
srun python predict_save_halo_cats.py 331
srun python predict_save_halo_cats.py 332
srun python predict_save_halo_cats.py 333
srun python predict_save_halo_cats.py 334
srun python predict_save_halo_cats.py 335
srun python predict_save_halo_cats.py 336
srun python predict_save_halo_cats.py 337
srun python predict_save_halo_cats.py 338
srun python predict_save_halo_cats.py 339
srun python predict_save_halo_cats.py 340
srun python predict_save_halo_cats.py 341
srun python predict_save_halo_cats.py 342
srun python predict_save_halo_cats.py 343
srun python predict_save_halo_cats.py 344
srun python predict_save_halo_cats.py 345
srun python predict_save_halo_cats.py 346
srun python predict_save_halo_cats.py 347
srun python predict_save_halo_cats.py 348
srun python predict_save_halo_cats.py 349
srun python predict_save_halo_cats.py 350
srun python predict_save_halo_cats.py 351
srun python predict_save_halo_cats.py 352
srun python predict_save_halo_cats.py 353
srun python predict_save_halo_cats.py 354
srun python predict_save_halo_cats.py 355
srun python predict_save_halo_cats.py 356
srun python predict_save_halo_cats.py 357
srun python predict_save_halo_cats.py 358
srun python predict_save_halo_cats.py 359
srun python predict_save_halo_cats.py 360
srun python predict_save_halo_cats.py 361
srun python predict_save_halo_cats.py 362
srun python predict_save_halo_cats.py 363
srun python predict_save_halo_cats.py 364
srun python predict_save_halo_cats.py 365
srun python predict_save_halo_cats.py 366
srun python predict_save_halo_cats.py 367
srun python predict_save_halo_cats.py 368
srun python predict_save_halo_cats.py 369
srun python predict_save_halo_cats.py 370
srun python predict_save_halo_cats.py 371
srun python predict_save_halo_cats.py 372
srun python predict_save_halo_cats.py 373
srun python predict_save_halo_cats.py 374
srun python predict_save_halo_cats.py 375
srun python predict_save_halo_cats.py 376
srun python predict_save_halo_cats.py 377
srun python predict_save_halo_cats.py 378
srun python predict_save_halo_cats.py 379
srun python predict_save_halo_cats.py 380
srun python predict_save_halo_cats.py 381
srun python predict_save_halo_cats.py 382
srun python predict_save_halo_cats.py 383
srun python predict_save_halo_cats.py 384
srun python predict_save_halo_cats.py 385
srun python predict_save_halo_cats.py 386
srun python predict_save_halo_cats.py 387
srun python predict_save_halo_cats.py 388
srun python predict_save_halo_cats.py 389
srun python predict_save_halo_cats.py 390
srun python predict_save_halo_cats.py 391
srun python predict_save_halo_cats.py 392
srun python predict_save_halo_cats.py 393
srun python predict_save_halo_cats.py 394
srun python predict_save_halo_cats.py 395
srun python predict_save_halo_cats.py 396
srun python predict_save_halo_cats.py 397
srun python predict_save_halo_cats.py 398
srun python predict_save_halo_cats.py 399
srun python predict_save_halo_cats.py 400
srun python predict_save_halo_cats.py 401
srun python predict_save_halo_cats.py 402
srun python predict_save_halo_cats.py 403
srun python predict_save_halo_cats.py 404
srun python predict_save_halo_cats.py 405
srun python predict_save_halo_cats.py 406
srun python predict_save_halo_cats.py 407
srun python predict_save_halo_cats.py 408
srun python predict_save_halo_cats.py 409
srun python predict_save_halo_cats.py 410
srun python predict_save_halo_cats.py 411
srun python predict_save_halo_cats.py 412
srun python predict_save_halo_cats.py 413
srun python predict_save_halo_cats.py 414
srun python predict_save_halo_cats.py 415
srun python predict_save_halo_cats.py 416
srun python predict_save_halo_cats.py 417
srun python predict_save_halo_cats.py 418
srun python predict_save_halo_cats.py 419
srun python predict_save_halo_cats.py 420
srun python predict_save_halo_cats.py 421
srun python predict_save_halo_cats.py 422
srun python predict_save_halo_cats.py 423
srun python predict_save_halo_cats.py 424
srun python predict_save_halo_cats.py 425
srun python predict_save_halo_cats.py 426
srun python predict_save_halo_cats.py 427
srun python predict_save_halo_cats.py 428
srun python predict_save_halo_cats.py 429
srun python predict_save_halo_cats.py 430
srun python predict_save_halo_cats.py 431
srun python predict_save_halo_cats.py 432
srun python predict_save_halo_cats.py 433
srun python predict_save_halo_cats.py 434
srun python predict_save_halo_cats.py 435
srun python predict_save_halo_cats.py 436
srun python predict_save_halo_cats.py 437
srun python predict_save_halo_cats.py 438
srun python predict_save_halo_cats.py 439
srun python predict_save_halo_cats.py 440
srun python predict_save_halo_cats.py 441
srun python predict_save_halo_cats.py 442
srun python predict_save_halo_cats.py 443
srun python predict_save_halo_cats.py 444
srun python predict_save_halo_cats.py 445
srun python predict_save_halo_cats.py 446
srun python predict_save_halo_cats.py 447
srun python predict_save_halo_cats.py 448
srun python predict_save_halo_cats.py 449
srun python predict_save_halo_cats.py 450
srun python predict_save_halo_cats.py 451
srun python predict_save_halo_cats.py 452
srun python predict_save_halo_cats.py 453
srun python predict_save_halo_cats.py 454
srun python predict_save_halo_cats.py 455
srun python predict_save_halo_cats.py 456
srun python predict_save_halo_cats.py 457
srun python predict_save_halo_cats.py 458
srun python predict_save_halo_cats.py 459
srun python predict_save_halo_cats.py 460
srun python predict_save_halo_cats.py 461
srun python predict_save_halo_cats.py 462
srun python predict_save_halo_cats.py 463
srun python predict_save_halo_cats.py 464
srun python predict_save_halo_cats.py 465
srun python predict_save_halo_cats.py 466
srun python predict_save_halo_cats.py 467
srun python predict_save_halo_cats.py 468
srun python predict_save_halo_cats.py 469
srun python predict_save_halo_cats.py 470
srun python predict_save_halo_cats.py 471
srun python predict_save_halo_cats.py 472
srun python predict_save_halo_cats.py 473
srun python predict_save_halo_cats.py 474
srun python predict_save_halo_cats.py 475
srun python predict_save_halo_cats.py 476
srun python predict_save_halo_cats.py 477
srun python predict_save_halo_cats.py 478
srun python predict_save_halo_cats.py 479
srun python predict_save_halo_cats.py 480
srun python predict_save_halo_cats.py 481
srun python predict_save_halo_cats.py 482
srun python predict_save_halo_cats.py 483
srun python predict_save_halo_cats.py 484
srun python predict_save_halo_cats.py 485
srun python predict_save_halo_cats.py 486
srun python predict_save_halo_cats.py 487
srun python predict_save_halo_cats.py 488
srun python predict_save_halo_cats.py 489
srun python predict_save_halo_cats.py 490
srun python predict_save_halo_cats.py 491
srun python predict_save_halo_cats.py 492
srun python predict_save_halo_cats.py 493
srun python predict_save_halo_cats.py 494
srun python predict_save_halo_cats.py 495
srun python predict_save_halo_cats.py 496
srun python predict_save_halo_cats.py 497
srun python predict_save_halo_cats.py 498
srun python predict_save_halo_cats.py 499
srun python predict_save_halo_cats.py 500
srun python predict_save_halo_cats.py 501
srun python predict_save_halo_cats.py 502
srun python predict_save_halo_cats.py 503
srun python predict_save_halo_cats.py 504
srun python predict_save_halo_cats.py 505
srun python predict_save_halo_cats.py 506
srun python predict_save_halo_cats.py 507
srun python predict_save_halo_cats.py 508
srun python predict_save_halo_cats.py 509
srun python predict_save_halo_cats.py 510
srun python predict_save_halo_cats.py 511
srun python predict_save_halo_cats.py 512
srun python predict_save_halo_cats.py 513
srun python predict_save_halo_cats.py 514
srun python predict_save_halo_cats.py 515
srun python predict_save_halo_cats.py 516
srun python predict_save_halo_cats.py 517
srun python predict_save_halo_cats.py 518
srun python predict_save_halo_cats.py 519
srun python predict_save_halo_cats.py 520
srun python predict_save_halo_cats.py 521
srun python predict_save_halo_cats.py 522
srun python predict_save_halo_cats.py 523
srun python predict_save_halo_cats.py 524
srun python predict_save_halo_cats.py 525
srun python predict_save_halo_cats.py 526
srun python predict_save_halo_cats.py 527
srun python predict_save_halo_cats.py 528
srun python predict_save_halo_cats.py 529
srun python predict_save_halo_cats.py 530
srun python predict_save_halo_cats.py 531
srun python predict_save_halo_cats.py 532
srun python predict_save_halo_cats.py 533
srun python predict_save_halo_cats.py 534
srun python predict_save_halo_cats.py 535
srun python predict_save_halo_cats.py 536
srun python predict_save_halo_cats.py 537
srun python predict_save_halo_cats.py 538
srun python predict_save_halo_cats.py 539
srun python predict_save_halo_cats.py 540
srun python predict_save_halo_cats.py 541
srun python predict_save_halo_cats.py 542
srun python predict_save_halo_cats.py 543
srun python predict_save_halo_cats.py 544
srun python predict_save_halo_cats.py 545
srun python predict_save_halo_cats.py 546
srun python predict_save_halo_cats.py 547
srun python predict_save_halo_cats.py 548
srun python predict_save_halo_cats.py 549
srun python predict_save_halo_cats.py 550
srun python predict_save_halo_cats.py 551
srun python predict_save_halo_cats.py 552
srun python predict_save_halo_cats.py 553
srun python predict_save_halo_cats.py 554
srun python predict_save_halo_cats.py 555
srun python predict_save_halo_cats.py 556
srun python predict_save_halo_cats.py 557
srun python predict_save_halo_cats.py 558
srun python predict_save_halo_cats.py 559
srun python predict_save_halo_cats.py 560
srun python predict_save_halo_cats.py 561
srun python predict_save_halo_cats.py 562
srun python predict_save_halo_cats.py 563
srun python predict_save_halo_cats.py 564
srun python predict_save_halo_cats.py 565
srun python predict_save_halo_cats.py 566
srun python predict_save_halo_cats.py 567
srun python predict_save_halo_cats.py 568
srun python predict_save_halo_cats.py 569
srun python predict_save_halo_cats.py 570
srun python predict_save_halo_cats.py 571
srun python predict_save_halo_cats.py 572
srun python predict_save_halo_cats.py 573
srun python predict_save_halo_cats.py 574
srun python predict_save_halo_cats.py 575
srun python predict_save_halo_cats.py 576
srun python predict_save_halo_cats.py 577
srun python predict_save_halo_cats.py 578
srun python predict_save_halo_cats.py 579
srun python predict_save_halo_cats.py 580
srun python predict_save_halo_cats.py 581
srun python predict_save_halo_cats.py 582
srun python predict_save_halo_cats.py 583
srun python predict_save_halo_cats.py 584
srun python predict_save_halo_cats.py 585
srun python predict_save_halo_cats.py 586
srun python predict_save_halo_cats.py 587
srun python predict_save_halo_cats.py 588
srun python predict_save_halo_cats.py 589
srun python predict_save_halo_cats.py 590
srun python predict_save_halo_cats.py 591
srun python predict_save_halo_cats.py 592
srun python predict_save_halo_cats.py 593
srun python predict_save_halo_cats.py 594
srun python predict_save_halo_cats.py 595
srun python predict_save_halo_cats.py 596
srun python predict_save_halo_cats.py 597
srun python predict_save_halo_cats.py 598
srun python predict_save_halo_cats.py 599


echo "All done!"
