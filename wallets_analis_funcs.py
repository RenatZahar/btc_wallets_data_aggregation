import os
import pandas as pd
import dask
from dask import dataframe as dd
from dask.distributed import Client
import numpy as np
from random import randint
import time 
import gc
import threading
import asyncio

lock_read = threading.Lock()
lock_save1 = threading.Lock()
lock_save2 = threading.Lock()
unique_wallets = []
is_processing = False
processing_lock = threading.Lock()

def make_unique_wallets_file_small():
    global lock_read
    global is_processing
    global processing_lock
    with lock_read:
        print(1)
        with processing_lock:
        # Проверяем, выполняется ли уже данный участок кода
            if is_processing:
                asyncio.sleep(0.3)
                return  # Если да, просто выходим из функции
            is_processing = True
            UNIQ_WALLETS_FILE_DIR = 'I:\\Traider_bot\\кошельки\\сбор данных по кошелькам\\unique_wallets_final.parquet'
            df = pd.read_parquet(UNIQ_WALLETS_FILE_DIR)
            df = df.reset_index(drop=True)
            df = df.iloc[:100]
            df.to_parquet(UNIQ_WALLETS_FILE_DIR)


def init(MEMORY_LIMIT, THREADS_PER_WORKER, WORKERS, DATA_FILE_DIR, NPARTITIONS):
    dask.config.set({'distributed.worker.memory.target': 0.1,
                 'distributed.worker.memory.spill': 0.35,
                 'distributed.worker.memory.pause': 0.75,
                 'distributed.worker.memory.terminate': 0.8,
                 'distributed.worker.timeout.interval': 500})

    
    client = Client(memory_limit=MEMORY_LIMIT, threads_per_worker=THREADS_PER_WORKER, n_workers=WORKERS, local_directory='I:/dask-temp')#, memory_limit=MEMORY_LIMIT, threads_per_worker=THREADS_PER_WORKER,  memory_limit='2GB', memory_limit='17GB', memory_limit='50GB',  threads_per_worker=1,
    client.run(lambda: gc.collect())
    print(client.dashboard_link)
    
    data_files = [os.path.join(DATA_FILE_DIR, file) for file in os.listdir(DATA_FILE_DIR) if file.endswith('.parquet')]
    columns = [
        'Wallet_id', 'Avg_btc_diff_price', 'Total_amount', 'Total_txs_per_wallet', 'Avg_action_days',
        'Avg_btc_sell_price', 'Avg_btc_buy_price',
        'Sell_txs_quant', 'Buy_txs_quant', 
        'Sell_amount', 'Buy_amount',
        'Avg_sell_days', 'Avg_buy_days'
            ]

    empty_pd_df = pd.DataFrame(columns=columns)
    wallets_df = dd.from_pandas(empty_pd_df, npartitions=NPARTITIONS)
    return data_files, wallets_df, empty_pd_df, client

def remove_duplicates(partition):
    return partition.drop_duplicates(subset='Wallet_id')

def get_unique_wallets(data_files, WALLETS_IN_ITERATION, NPARTITIONS, PARTITION_SIZE):
    global lock_read
    global unique_wallets
    with lock_read:
        file_path = f'I:\\Traider_bot\\кошельки\\сбор данных по кошелькам\\unique_wallets_final.parquet'
        if not os.path.exists(file_path):
            ddf = dd.read_parquet(data_files, engine='fastparquet', columns=['Wallet_id'])
            ddf = ddf.repartition(partition_size=PARTITION_SIZE)
            unique_wallets = ddf.map_partitions(remove_duplicates).compute()
            unique_wallets = ddf['Wallet_id'].drop_duplicates().compute()
            unique_wallets = unique_wallets.reset_index(drop=True)
            unique_wallets = unique_wallets.to_frame(name='Wallet_id')
            unique_wallets.to_parquet(file_path, engine='fastparquet')

        if not unique_wallets:
            unique_wallets = pd.read_parquet(file_path)

        unique_wallets = unique_wallets.reset_index(drop=True)
        unique_wallets_list = unique_wallets['Wallet_id']  # Выбор столбца как Series
        unique_wallets_list = unique_wallets_list.tolist()  # Преобразование в список
        print(unique_wallets_list[:50])
        print(f'Всего необработанных кошельков: {len(unique_wallets_list)}')
        batches = [unique_wallets_list[i:i + WALLETS_IN_ITERATION] for i in range(0, len(unique_wallets_list), WALLETS_IN_ITERATION)]
        return batches



def filter_wallets(df, wallets):
    return df[df['Wallet_id'].isin(wallets)]

def process_batch(wallets_batch, txs_ddf, FILES_IN_ITERATION, NPARTITIONS, PARTITION_SIZE, WORKERS):
    group_wallets_info = process_wallet_group(wallets_batch, txs_ddf, FILES_IN_ITERATION, NPARTITIONS, PARTITION_SIZE, WORKERS)
    # Дополнительные операции, если они необходимы
    return group_wallets_info

def process_wallets_group(wallets_group, transactions_ddf):
    group_ddf = transactions_ddf[transactions_ddf['Wallet_id'].isin(wallets_group)]
    return get_wallets_info(group_ddf)

def process_wallet_group(wallets, ddf, FILES_IN_ITERATION, NPARTITIONS, PARTITION_SIZE, WORKERS):
    filtered_ddf = filter_wallets(ddf, wallets)
    filtered_ddf = filtered_ddf.repartition(partition_size=PARTITION_SIZE)
    wallets_groups = np.array_split(wallets, WORKERS)
    
    wallets_groups = [group.tolist() for group in wallets_groups]
    results = [dask.delayed(process_wallets_group_ddf)(group, filtered_ddf) for group in wallets_groups]
    final_result_ddf = dask.compute(*results)
    final_wallets_stats_df = pd.concat(final_result_ddf)
    return final_wallets_stats_df

def process_wallets_group_ddf(wallets, transactions_ddf):
    filtered_ddf = transactions_ddf[transactions_ddf['Wallet_id'].isin(wallets)]
    wallet_info_ddf = get_wallets_info(filtered_ddf)
    return wallet_info_ddf

def get_wallets_info(transactions_ddf):

    print('Агрегация данных по кошелькам')
    transactions_ddf = transactions_ddf.groupby(['Transaction_id', 'Wallet_id', 'Btc_block_time_price', 'Block_time']).agg({'Amount': 'sum', 'Transactions_Count': 'sum'}).reset_index()

    transactions_ddf.loc[:, 'Date'] = dd.to_datetime(transactions_ddf['Block_time'], unit='s')
    transactions_ddf = transactions_ddf.sort_values(by=['Wallet_id', 'Date'])

    transactions_ddf['Time_diff'] = transactions_ddf.groupby('Wallet_id')['Date'].diff()
    transactions_ddf['Time_diff_days'] = transactions_ddf['Time_diff'].dt.total_seconds() / (24 * 3600)
    # Разделяем транзакции на продажи и покупки
    sells = transactions_ddf[transactions_ddf['Amount'] < 0]
    buys = transactions_ddf[transactions_ddf['Amount'] > 0]

    # Агрегация данных по кошелькам
    wallets_stats = transactions_ddf.groupby('Wallet_id').agg({
        'Transactions_Count': 'sum',
        'Amount': 'sum',
        'Time_diff_days': 'mean',
        'Btc_block_time_price': 'mean'
    })
    print(wallets_stats.head())
    # Вычисление дополнительных статистик
    wallets_stats['Avg_sell_days'] = sells.groupby('Wallet_id')['Time_diff_days'].mean().round(2)
    wallets_stats['Avg_buy_days'] = buys.groupby('Wallet_id')['Time_diff_days'].mean().round(2)
    wallets_stats['Avg_action_days'] = transactions_ddf.groupby('Wallet_id')['Time_diff_days'].mean().round(2)
    wallets_stats['Avg_btc_sell_price'] = sells.groupby('Wallet_id')['Btc_block_time_price'].mean()
    wallets_stats['Avg_btc_buy_price'] = buys.groupby('Wallet_id')['Btc_block_time_price'].mean()
    wallets_stats['Avg_btc_diff_price'] = wallets_stats['Avg_btc_sell_price'] - wallets_stats['Avg_btc_buy_price']
    wallets_stats['Sell_txs_quant'] = sells.groupby('Wallet_id').size().astype(int)
    wallets_stats['Buy_txs_quant'] = buys.groupby('Wallet_id').size().astype(int)
    wallets_stats['Total_txs_per_wallet'] = wallets_stats['Sell_txs_quant'] + wallets_stats['Buy_txs_quant']
    wallets_stats['Sell_amount'] = sells.groupby('Wallet_id')['Amount'].sum()
    wallets_stats['Buy_amount'] = buys.groupby('Wallet_id')['Amount'].sum()
    wallets_stats['Total_amount'] = wallets_stats['Sell_amount'].abs() + wallets_stats['Buy_amount']
    wallets_stats = wallets_stats.reset_index()
    return wallets_stats

def save_wallets_tp_parquet(wallets_df, empty_pd_df, wallets_list, client, iteration_index, NPARTITIONS, WALLETS_FILE_DIR):
    global lock_save1
    with lock_save1:
        wallets_df = wallets_df.reset_index(drop=True)
        print('Сохранение wallets')
        wallets_df = client.persist(wallets_df)
        wallets_df = wallets_df.compute()
        x = randint(0,100000)
        wallets_df.to_parquet(
            f'{WALLETS_FILE_DIR}\\wallets_data_iteretion_{iteration_index}_{x}.parquet',
            engine='fastparquet')
        print(wallets_df.head(10))
        print(wallets_df.info())
        print('Конец сохранения')

        remove_processed_wallets(wallets_list, WALLETS_FILE_DIR, client)
    
def remove_processed_wallets(wallets_list, WALLETS_FILE_DIR, client, file_path='I:\\Traider_bot\\кошельки\\сбор данных по кошелькам\\unique_wallets_final.parquet'):
    global lock_save2
    with lock_save2:
        print('Удаление обработанных кошельков')
        if os.path.exists(file_path):
            unique_wallets_df = pd.read_parquet(file_path, engine='fastparquet')
            print(f'До удаления, кол-во кошельков: {len(unique_wallets_df)}')
            filtered_df = unique_wallets_df[~unique_wallets_df['Wallet_id'].isin(wallets_list)]
            print(f'После {len(filtered_df)}')

            if filtered_df.index.size > 0:
                print('Сохранение обновленного DataFrame')
                os.remove(file_path)
                filtered_df.to_parquet(file_path)
            else:
                time.sleep(5)
                client.close()
                os.remove(file_path)
                print('DataFrame пуст, файл будет удален, файлы будут собраны в 1 дф')
                data_files = [os.path.join(WALLETS_FILE_DIR, file) for file in os.listdir(WALLETS_FILE_DIR) if file.endswith('.parquet')]
                all_wallets_df = pd.DataFrame()
                
                for i in data_files:
                    df = pd.read_parquet(i)
                    all_wallets_df = pd.concat([all_wallets_df, df])
                all_wallets_df = all_wallets_df.reset_index(drop=True)
                all_wallets_df.to_parquet(WALLETS_FILE_DIR+'\\all_wallets_df.parquet')
                print('Все кошельки скомплементированы')
                for file in os.listdir(WALLETS_FILE_DIR):
                    file_path = os.path.join(WALLETS_FILE_DIR, file)
                    if file != 'all_wallets_df.parquet' and file.endswith('.parquet'):
                        os.remove(file_path)
                        print(f'Файл {file_path} удален')

        else:
            print(f'Файл {file_path} не найден.')

