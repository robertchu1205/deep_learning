# Docker
## Gitlab-runner setup
### Purpose
CICD in gitlab, BUILD - TEST - UPLOAD TO HABOUR
### Detail
```
# Run container fist, a runner can be running for several projects
docker run -d --name {container-name} --restart=always -v /var/run/docker.sock:/var/run/docker.sock -v /srv/robert-runner/config:/etc/gitlab-runner gitlab-runner:alpine-{version}
# Register by docker exec, gitlab-url should start from http or https
docker exec -it {container-name} gitlab-runner register -n --url {gitlab-url} --registration-token {token-found-from-project-in-gitlab-setting-CICD} --clone-url {gitlab-url} --executor docker --docker-image "docker:latest" --docker-privileged
```