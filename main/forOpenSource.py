import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm



def load_data(file_path):
    return pd.read_csv(file_path)

def display_basic_info(data):
    # Print the first 5 rows
    print("First 5 rows of the DataFrame:")
    print(data.head(5))
    print()

    # Print DataFrame information
    print("DataFrame information:")
    print(data.info())
    print()

    # Print summary statistics
    print("Summary statistics:")
    print(data.describe())
    print()

    # Print DataFrame shape
    print("DataFrame shape:")
    print(data.shape)
    print()

    # Print missing values count
    print("Missing values count:")
    missing_count = data.isna().sum()
    print(missing_count)
    print()

def extract_date_features(data):
    """
    Extract date features from the 'event_time' column.

    Args:
    - data (DataFrame): Input DataFrame.
    """

    data['event_time'] = pd.to_datetime(data['event_time'])
    data['year'], data['month'], data['day'] = data['event_time'].dt.year,  data['event_time'].dt.month,  data['event_time'].dt.day
    data['hour'] = data['event_time'].dt.hour

    '''
    print("Year, Month, and Day columns:")
    print(data[['year','month', 'day']])
    print()
    '''


def visualize_event_counts_hourly(data):
    """
    Visualize event counts hourly.

    Args:
    - data (DataFrame): Input DataFrame.
    """
    event_types = data['event_type'].unique()
    hourly_counts = data.groupby(['hour', 'event_type']).size().unstack(fill_value=0)

    fig, axes = plt.subplots(nrows=len(event_types), ncols=1, figsize=(12, 18), sharex=True)

    for i, event_type in enumerate(event_types):
        axes[i].bar(hourly_counts.index, hourly_counts[event_type], color='blue', alpha=0.7)
        axes[i].set_title(f'Number of {event_type.capitalize()} Events per Hour')
        axes[i].set_ylabel('Number of Events')
        axes[i].set_xticks(range(0, 24))

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3)
    plt.show()

def visualize_event_counts_monthly(data):
    """
    Visualize event counts monthly.

    Args:
    - data (DataFrame): Input DataFrame.
    """
    event_types = data['event_type'].unique()
    monthly_counts = data.groupby(['month', 'event_type']).size().unstack(fill_value=0)

    fig, axes = plt.subplots(nrows=len(event_types), ncols=1, figsize=(12, 18), sharex=True)

    temp = range(0, monthly_counts.shape[0])
    for i, event_type in enumerate(event_types):
        axes[i].bar(temp, monthly_counts[event_type], color='green', alpha=0.7)
        axes[i].set_title(f'Number of {event_type.capitalize()} Events per Month')
        axes[i].set_ylabel('Number of Events')
        axes[i].set_xticks(temp)
        axes[i].set_xticklabels(['1', '2', '9', '10', '11', '12'])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3)
    plt.show()

def visualize_event_type(df):
    # event_type에 따른 행의 개수 계산
    event_counts = df['event_type'].value_counts()

    # 히스토그램 생성
    event_counts.plot(kind='bar', edgecolor='black')

    # 그래프 제목과 축 레이블 설정
    plt.title('Event Type Count')
    plt.xlabel('Event Type')
    plt.ylabel('Count')

    plt.xticks(rotation=0)
    # 그래프 표시
    plt.show()

def KFold(knn,x,y,n):
    # Train model with cv 5
    cv_scores=cross_val_score(knn, x,y,cv=5)

    # Print each cv score & average
    print('cv_scores: ',cv_scores)
    print('cv_scores mean: {}'.format(np.mean(cv_scores)))

def fill_category_code(df):
    # null이 아닌 행만 사용하여 데이터 분할
    not_null_data = df[~df['category_code'].isnull()]
    X = not_null_data[['category_id', 'price']]
    y = not_null_data['category_code']

    # 데이터를 학습 및 테스트 세트로 분할 (hold-out method)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN 모델 학습
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # KNN 모델 다시 학습
    knn_model.fit(X_train, y_train)

    # 테스트 세트에 대한 예측
    y_pred = knn_model.predict(X_test)

    # category_code의 null 값을 예측하여 채움
    X_missing = df[df['category_code'].isnull()][['category_id', 'price']]
    predicted_category_codes = knn_model.predict(X_missing)
    df.loc[df['category_code'].isnull(), 'category_code'] = predicted_category_codes

    # kfold로 성능 체크
    KFold(knn_model, X,y,3)

def fill_brand(df):
    # null이 아닌 행만 사용하여 데이터 분할
    not_null_data = df[~df['brand'].isnull()]
    X = not_null_data[['category_id', 'price', 'product_id']]
    y = not_null_data['brand']

    # 데이터를 학습 및 테스트 세트로 분할 (hold-out method)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN 모델 학습
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # KNN 모델 다시 학습
    knn_model.fit(X_train, y_train)

    # 테스트 세트에 대한 예측
    y_pred = knn_model.predict(X_test)

    # category_code의 null 값을 예측하여 채움
    X_missing = df[df['brand'].isnull()][['category_id', 'price', 'product_id']]
    predicted_brand = knn_model.predict(X_missing)
    df.loc[df['brand'].isnull(), 'brand'] = predicted_brand

    # kfold로 성능 체크
    KFold(knn_model, X,y,3)

def label_encode_columns(df):
    label_encoder = LabelEncoder()

    # 'event_type', 'category_code', 'brand' 열을 label encoding
    df['event_type'] = label_encoder.fit_transform(df['event_type'])
    df['category_code'] = label_encoder.fit_transform(df['category_code'].astype(str))
    df['brand'] = label_encoder.fit_transform(df['brand'].astype(str))


def standard(df, column_name):
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])

def minMax(df, column_name):
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])

def robust(df, column_name):
    scaler = RobustScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])


def find_optimal_clusters(data):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    return sse

def visualize_elbow_method(sse):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()


def scaler(identifier, new_df):
    '''
    identifier 0 -> StandardScaler()
    identifier 1 -> MinMaxScaler()
    identifier 2 -> RobustScaler()
    '''
    normalize_columns = ['price', 'days_since_last_event', 'event_count', 'total_purchase_amount']
    for column in normalize_columns:
        if identifier == 0:      standard(new_df, column)
        elif identifier==1:    minMax(new_df, column)
        else:   robust(new_df, column)


def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans.labels_

def visualize_clusters(data, labels, n_clusters):
    plt.figure()
    colors = ['r', 'g', 'b','c', 'm']
    for cluster in range(n_clusters):
        cluster_data = data[labels == cluster]
        plt.scatter(cluster_data['days_since_last_event'], cluster_data['total_purchase_amount'],
                    c=colors[cluster], label=f'Cluster {cluster}', alpha=0.3)
    plt.xlabel('Days Since Last Event')
    plt.ylabel('Total Purchase Amount')
    plt.legend()
    plt.show()

def runKmeans(n_clusters, new_df):
    labels = apply_kmeans(new_df[['days_since_last_event', 'total_purchase_amount']], n_clusters)
    visualize_clusters(new_df, labels, n_clusters)

#Hierarchical Agglomerative Clustering
#single Linkage
def runAgglomerative(data):
    subset = data.sample(n=500, random_state=42) if len(data) > 500 else data  # Use a subset of the data
    linked = linkage(subset, 'single')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', show_leaf_counts=True)
    plt.show()




def apply_Agglomerative(data, n_clusters):
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    labels = cluster.fit_predict(data)
    return labels


def main():

    cluster_score=[]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 30)

    file_path = '/content/events.csv'
    df = load_data(file_path)

    display_basic_info(df)

    extract_date_features(df)


    visualize_event_counts_hourly(df)
    visualize_event_counts_monthly(df)
    visualize_event_type(df)


    df['event_time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df_sorted = df.sort_values(by=['user_id', 'event_time'], ascending=[True, False])  # 각 user_id 별로 최신 이벤트를 찾기 위해 정렬
    latest_events = df_sorted.drop_duplicates(subset=['user_id'], keep='first').copy()  # 최신 이벤트만

    current_date = pd.to_datetime(datetime.now())  # 현재 날짜를 Timestamp 형식으로 변환하여 사용
    latest_events['days_since_last_event'] = (current_date - latest_events['event_time']).dt.days  # 차이 구하기

    event_counts = df['user_id'].value_counts().reset_index()
    event_counts.columns = ['user_id', 'event_count']  # 각 user_id별 event 발생 횟수 계산

    purchase_sums = df[df['event_type'] == 'purchase'].groupby('user_id')['price'].sum().reset_index()
    purchase_sums.columns = ['user_id', 'total_purchase_amount']  # 구매 합산

    new_df = pd.merge(latest_events, event_counts, on='user_id')
    new_df = pd.merge(new_df, purchase_sums, on='user_id', how='left')

    new_df = new_df.drop(columns=['event_time'])
    new_df = new_df.drop(columns=['user_session'])

    fill_category_code(new_df)
    fill_brand(new_df)

    label_encode_columns(new_df)

    new_df = new_df.fillna(0)  # purchase가 NaN 값인 데이터 0으로 변경

    '''
    ==============================================================================================================
    '''
    # scaling
    for i in  range(0,3):

        scaler(i, new_df)


        new_df = new_df[(new_df['price'] >= -3) & (new_df['price'] <= 3)]
        new_df = new_df[(new_df['days_since_last_event'] >= -3) & (new_df['days_since_last_event'] <= 3)]
        new_df = new_df[(new_df['event_count'] >= -3) & (new_df['event_count'] <= 3)]



        sse = find_optimal_clusters(new_df[['days_since_last_event', 'total_purchase_amount']])
        visualize_elbow_method(sse)

        # Hierarchical Agglomerative Clustering
        runAgglomerative(new_df[['days_since_last_event', 'total_purchase_amount']])

        for n_clusters in range(2,6):
            '''
            클러스터링은 원본 데이터로 하는데, 
            실루엣 점수는 너무 오래 걸리니까 1만개만 샘플링해서 계산
            '''
            # K-means
            runKmeans(n_clusters, new_df)

            # 랜덤으로 추출
            sampled_df = new_df.sample(n=10000, replace=False)

            # 점수 출력 (Kmeans)
            Kmeans_labels = apply_kmeans(sampled_df[['days_since_last_event', 'total_purchase_amount']], n_clusters=n_clusters)
            Kmeans_silhouette_avg = silhouette_score(sampled_df[['days_since_last_event', 'total_purchase_amount']], Kmeans_labels)

            print(f' Kmeans When n_clusters: {n_clusters} silhouette_avg: {Kmeans_silhouette_avg}')
            # 점수 저장
            data={}
            if i==0:  data['scaler']='StandardScaler'
            elif i==1:  data['scaler']='MinMaxScaler'
            else: data['scaler']='RobustScaler'
            data['algorithm']='kmeans'
            data['n_clusters']=n_clusters
            data['score']=Kmeans_silhouette_avg
            cluster_score.append(data)

            # 점수 출력 (Agglomerative clustering)
            Agglomerative_labels=apply_Agglomerative(sampled_df[['days_since_last_event', 'total_purchase_amount']], n_clusters=n_clusters)
            Agg_silhouette_avg=silhouette_score(sampled_df[['days_since_last_event', 'total_purchase_amount']], Agglomerative_labels)

            print(f'Agglomerative When n_clusters: {n_clusters} silhouette_avg: {Agg_silhouette_avg}')
            # 점수 저장
            data={}
            if i==0:  data['scaler']='StandardScaler'
            elif i==1:  data['scaler']='MinMaxScaler'
            else: data['scaler']='RobustScaler'
            data['algorithm']='Agglomerative'
            data['n_clusters']=n_clusters
            data['score']=Agg_silhouette_avg
            cluster_score.append(data)


        # 시각화
        cluster_lists = [2, 3, 4, 5]
        n_cols = len(cluster_lists)

        X_features = sampled_df[['days_since_last_event', 'total_purchase_amount']]

        # Subplot setup
        fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

        for ind, n_cluster in enumerate(cluster_lists):  # n이 2, 3, 4, 5일 때
            clusterer = KMeans(n_clusters=n_cluster)
            cluster_labels = clusterer.fit_predict(X_features)
            sil_avg = silhouette_score(X_features, cluster_labels)
            sil_values = silhouette_samples(X_features, cluster_labels)

            y_lower = 10
            axs[ind].set_title('Number of Clusters: ' + str(n_cluster) + '\n' \
                                                                         'Silhouette Score: ' + str(round(sil_avg, 3)))
            axs[ind].set_xlabel("The silhouette coefficient values")
            axs[ind].set_ylabel("Cluster label")
            axs[ind].set_xlim([-0.1, 1])
            axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
            axs[ind].set_yticks([])
            axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            # Fill betweenx for silhouette plots
            for i in range(n_cluster):
                ith_cluster_sil_values = sil_values[cluster_labels == i]
                ith_cluster_sil_values.sort()

                size_cluster_i = ith_cluster_sil_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_cluster)
                axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                       facecolor=color, edgecolor=color, alpha=0.7)
                axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

        plt.tight_layout()
        plt.show()

    # score를 기준으로 데이터 정렬
    sorted_data = sorted(cluster_score, key=lambda x: x['score'], reverse=True)

    # 상위 5개 데이터 추출
    top_5_data = sorted_data[:5]

    print('Top 5 combination')
    # 상위 5개 데이터의 각 key, value 출력
    for item in top_5_data:
        print("Data:")
        for key, value in item.items():
            print(f"{key}: {value}")
        print()  # 줄 바꿈



if __name__ == "__main__":
    main()
