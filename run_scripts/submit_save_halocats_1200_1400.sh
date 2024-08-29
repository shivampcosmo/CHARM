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


srun --time=1 python predict_save_halo_cats.py 1200
srun --time=1 python predict_save_halo_cats.py 1201
srun --time=1 python predict_save_halo_cats.py 1202
srun --time=1 python predict_save_halo_cats.py 1203
srun --time=1 python predict_save_halo_cats.py 1204
srun --time=1 python predict_save_halo_cats.py 1205
srun --time=1 python predict_save_halo_cats.py 1206
srun --time=1 python predict_save_halo_cats.py 1207
srun --time=1 python predict_save_halo_cats.py 1208
srun --time=1 python predict_save_halo_cats.py 1209
srun --time=1 python predict_save_halo_cats.py 1210
srun --time=1 python predict_save_halo_cats.py 1211
srun --time=1 python predict_save_halo_cats.py 1212
srun --time=1 python predict_save_halo_cats.py 1213
srun --time=1 python predict_save_halo_cats.py 1214
srun --time=1 python predict_save_halo_cats.py 1215
srun --time=1 python predict_save_halo_cats.py 1216
srun --time=1 python predict_save_halo_cats.py 1217
srun --time=1 python predict_save_halo_cats.py 1218
srun --time=1 python predict_save_halo_cats.py 1219
srun --time=1 python predict_save_halo_cats.py 1220
srun --time=1 python predict_save_halo_cats.py 1221
srun --time=1 python predict_save_halo_cats.py 1222
srun --time=1 python predict_save_halo_cats.py 1223
srun --time=1 python predict_save_halo_cats.py 1224
srun --time=1 python predict_save_halo_cats.py 1225
srun --time=1 python predict_save_halo_cats.py 1226
srun --time=1 python predict_save_halo_cats.py 1227
srun --time=1 python predict_save_halo_cats.py 1228
srun --time=1 python predict_save_halo_cats.py 1229
srun --time=1 python predict_save_halo_cats.py 1230
srun --time=1 python predict_save_halo_cats.py 1231
srun --time=1 python predict_save_halo_cats.py 1232
srun --time=1 python predict_save_halo_cats.py 1233
srun --time=1 python predict_save_halo_cats.py 1234
srun --time=1 python predict_save_halo_cats.py 1235
srun --time=1 python predict_save_halo_cats.py 1236
srun --time=1 python predict_save_halo_cats.py 1237
srun --time=1 python predict_save_halo_cats.py 1238
srun --time=1 python predict_save_halo_cats.py 1239
srun --time=1 python predict_save_halo_cats.py 1240
srun --time=1 python predict_save_halo_cats.py 1241
srun --time=1 python predict_save_halo_cats.py 1242
srun --time=1 python predict_save_halo_cats.py 1243
srun --time=1 python predict_save_halo_cats.py 1244
srun --time=1 python predict_save_halo_cats.py 1245
srun --time=1 python predict_save_halo_cats.py 1246
srun --time=1 python predict_save_halo_cats.py 1247
srun --time=1 python predict_save_halo_cats.py 1248
srun --time=1 python predict_save_halo_cats.py 1249
srun --time=1 python predict_save_halo_cats.py 1250
srun --time=1 python predict_save_halo_cats.py 1251
srun --time=1 python predict_save_halo_cats.py 1252
srun --time=1 python predict_save_halo_cats.py 1253
srun --time=1 python predict_save_halo_cats.py 1254
srun --time=1 python predict_save_halo_cats.py 1255
srun --time=1 python predict_save_halo_cats.py 1256
srun --time=1 python predict_save_halo_cats.py 1257
srun --time=1 python predict_save_halo_cats.py 1258
srun --time=1 python predict_save_halo_cats.py 1259
srun --time=1 python predict_save_halo_cats.py 1260
srun --time=1 python predict_save_halo_cats.py 1261
srun --time=1 python predict_save_halo_cats.py 1262
srun --time=1 python predict_save_halo_cats.py 1263
srun --time=1 python predict_save_halo_cats.py 1264
srun --time=1 python predict_save_halo_cats.py 1265
srun --time=1 python predict_save_halo_cats.py 1266
srun --time=1 python predict_save_halo_cats.py 1267
srun --time=1 python predict_save_halo_cats.py 1268
srun --time=1 python predict_save_halo_cats.py 1269
srun --time=1 python predict_save_halo_cats.py 1270
srun --time=1 python predict_save_halo_cats.py 1271
srun --time=1 python predict_save_halo_cats.py 1272
srun --time=1 python predict_save_halo_cats.py 1273
srun --time=1 python predict_save_halo_cats.py 1274
srun --time=1 python predict_save_halo_cats.py 1275
srun --time=1 python predict_save_halo_cats.py 1276
srun --time=1 python predict_save_halo_cats.py 1277
srun --time=1 python predict_save_halo_cats.py 1278
srun --time=1 python predict_save_halo_cats.py 1279
srun --time=1 python predict_save_halo_cats.py 1280
srun --time=1 python predict_save_halo_cats.py 1281
srun --time=1 python predict_save_halo_cats.py 1282
srun --time=1 python predict_save_halo_cats.py 1283
srun --time=1 python predict_save_halo_cats.py 1284
srun --time=1 python predict_save_halo_cats.py 1285
srun --time=1 python predict_save_halo_cats.py 1286
srun --time=1 python predict_save_halo_cats.py 1287
srun --time=1 python predict_save_halo_cats.py 1288
srun --time=1 python predict_save_halo_cats.py 1289
srun --time=1 python predict_save_halo_cats.py 1290
srun --time=1 python predict_save_halo_cats.py 1291
srun --time=1 python predict_save_halo_cats.py 1292
srun --time=1 python predict_save_halo_cats.py 1293
srun --time=1 python predict_save_halo_cats.py 1294
srun --time=1 python predict_save_halo_cats.py 1295
srun --time=1 python predict_save_halo_cats.py 1296
srun --time=1 python predict_save_halo_cats.py 1297
srun --time=1 python predict_save_halo_cats.py 1298
srun --time=1 python predict_save_halo_cats.py 1299
srun --time=1 python predict_save_halo_cats.py 1300
srun --time=1 python predict_save_halo_cats.py 1301
srun --time=1 python predict_save_halo_cats.py 1302
srun --time=1 python predict_save_halo_cats.py 1303
srun --time=1 python predict_save_halo_cats.py 1304
srun --time=1 python predict_save_halo_cats.py 1305
srun --time=1 python predict_save_halo_cats.py 1306
srun --time=1 python predict_save_halo_cats.py 1307
srun --time=1 python predict_save_halo_cats.py 1308
srun --time=1 python predict_save_halo_cats.py 1309
srun --time=1 python predict_save_halo_cats.py 1310
srun --time=1 python predict_save_halo_cats.py 1311
srun --time=1 python predict_save_halo_cats.py 1312
srun --time=1 python predict_save_halo_cats.py 1313
srun --time=1 python predict_save_halo_cats.py 1314
srun --time=1 python predict_save_halo_cats.py 1315
srun --time=1 python predict_save_halo_cats.py 1316
srun --time=1 python predict_save_halo_cats.py 1317
srun --time=1 python predict_save_halo_cats.py 1318
srun --time=1 python predict_save_halo_cats.py 1319
srun --time=1 python predict_save_halo_cats.py 1320
srun --time=1 python predict_save_halo_cats.py 1321
srun --time=1 python predict_save_halo_cats.py 1322
srun --time=1 python predict_save_halo_cats.py 1323
srun --time=1 python predict_save_halo_cats.py 1324
srun --time=1 python predict_save_halo_cats.py 1325
srun --time=1 python predict_save_halo_cats.py 1326
srun --time=1 python predict_save_halo_cats.py 1327
srun --time=1 python predict_save_halo_cats.py 1328
srun --time=1 python predict_save_halo_cats.py 1329
srun --time=1 python predict_save_halo_cats.py 1330
srun --time=1 python predict_save_halo_cats.py 1331
srun --time=1 python predict_save_halo_cats.py 1332
srun --time=1 python predict_save_halo_cats.py 1333
srun --time=1 python predict_save_halo_cats.py 1334
srun --time=1 python predict_save_halo_cats.py 1335
srun --time=1 python predict_save_halo_cats.py 1336
srun --time=1 python predict_save_halo_cats.py 1337
srun --time=1 python predict_save_halo_cats.py 1338
srun --time=1 python predict_save_halo_cats.py 1339
srun --time=1 python predict_save_halo_cats.py 1340
srun --time=1 python predict_save_halo_cats.py 1341
srun --time=1 python predict_save_halo_cats.py 1342
srun --time=1 python predict_save_halo_cats.py 1343
srun --time=1 python predict_save_halo_cats.py 1344
srun --time=1 python predict_save_halo_cats.py 1345
srun --time=1 python predict_save_halo_cats.py 1346
srun --time=1 python predict_save_halo_cats.py 1347
srun --time=1 python predict_save_halo_cats.py 1348
srun --time=1 python predict_save_halo_cats.py 1349
srun --time=1 python predict_save_halo_cats.py 1350
srun --time=1 python predict_save_halo_cats.py 1351
srun --time=1 python predict_save_halo_cats.py 1352
srun --time=1 python predict_save_halo_cats.py 1353
srun --time=1 python predict_save_halo_cats.py 1354
srun --time=1 python predict_save_halo_cats.py 1355
srun --time=1 python predict_save_halo_cats.py 1356
srun --time=1 python predict_save_halo_cats.py 1357
srun --time=1 python predict_save_halo_cats.py 1358
srun --time=1 python predict_save_halo_cats.py 1359
srun --time=1 python predict_save_halo_cats.py 1360
srun --time=1 python predict_save_halo_cats.py 1361
srun --time=1 python predict_save_halo_cats.py 1362
srun --time=1 python predict_save_halo_cats.py 1363
srun --time=1 python predict_save_halo_cats.py 1364
srun --time=1 python predict_save_halo_cats.py 1365
srun --time=1 python predict_save_halo_cats.py 1366
srun --time=1 python predict_save_halo_cats.py 1367
srun --time=1 python predict_save_halo_cats.py 1368
srun --time=1 python predict_save_halo_cats.py 1369
srun --time=1 python predict_save_halo_cats.py 1370
srun --time=1 python predict_save_halo_cats.py 1371
srun --time=1 python predict_save_halo_cats.py 1372
srun --time=1 python predict_save_halo_cats.py 1373
srun --time=1 python predict_save_halo_cats.py 1374
srun --time=1 python predict_save_halo_cats.py 1375
srun --time=1 python predict_save_halo_cats.py 1376
srun --time=1 python predict_save_halo_cats.py 1377
srun --time=1 python predict_save_halo_cats.py 1378
srun --time=1 python predict_save_halo_cats.py 1379
srun --time=1 python predict_save_halo_cats.py 1380
srun --time=1 python predict_save_halo_cats.py 1381
srun --time=1 python predict_save_halo_cats.py 1382
srun --time=1 python predict_save_halo_cats.py 1383
srun --time=1 python predict_save_halo_cats.py 1384
srun --time=1 python predict_save_halo_cats.py 1385
srun --time=1 python predict_save_halo_cats.py 1386
srun --time=1 python predict_save_halo_cats.py 1387
srun --time=1 python predict_save_halo_cats.py 1388
srun --time=1 python predict_save_halo_cats.py 1389
srun --time=1 python predict_save_halo_cats.py 1390
srun --time=1 python predict_save_halo_cats.py 1391
srun --time=1 python predict_save_halo_cats.py 1392
srun --time=1 python predict_save_halo_cats.py 1393
srun --time=1 python predict_save_halo_cats.py 1394
srun --time=1 python predict_save_halo_cats.py 1395
srun --time=1 python predict_save_halo_cats.py 1396
srun --time=1 python predict_save_halo_cats.py 1397
srun --time=1 python predict_save_halo_cats.py 1398
srun --time=1 python predict_save_halo_cats.py 1399


echo "All done!"
