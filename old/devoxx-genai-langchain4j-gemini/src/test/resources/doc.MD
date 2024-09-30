
### Setup Google

- Authenticate with the browser

```console
gcloud auth application-default login  
```

- List Projects

```console
gcloud projects list
```

> Output
>
> ```console
> PROJECT_ID            NAME                  PROJECT_NUMBER
> datastax-gcp-mp-dev   datastax-gcp-mp-dev   345489893840
> datastax-gcp-mp-test  datastax-gcp-mp-test  90574246729
> devoxxfrance          devoxxfrance          55199250840
> ```

- Set Project

```console
gcloud config set project 55199250840
```

- Set the quota for the project
```console
gcloud auth application-default set-quota-project 55199250840 
gcloud compute project-info describe --project devoxxfrance
```

