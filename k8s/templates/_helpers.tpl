{{/*
Resolve the Secret name: a pre-existing (externally managed) Secret when
secrets.existingSecret is set, otherwise the chart-created Secret.
*/}}
{{- define "qufin.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- printf "%s-secrets" .Release.Name -}}
{{- end -}}
{{- end -}}

{{/*
Resolve the fully-qualified image reference, preferring a digest pin over the
mutable tag when image.digest is provided.
*/}}
{{- define "qufin.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end -}}

{{/*
Credentialed Redis broker/result env. Assembled from the ConfigMap host/port and
the Secret-sourced REDIS_PASSWORD via Kubernetes $(VAR) interpolation, so the
password is never written into the ConfigMap. db0=cache, db1=broker, db2=results.
REDIS_PASSWORD must be declared earlier in the same container's env list.
*/}}
{{- define "qufin.redisEnv" -}}
- name: REDIS_URL
  value: "redis://:$(REDIS_PASSWORD)@$(REDIS_HOST):$(REDIS_PORT)/0"
- name: CELERY_BROKER_URL
  value: "redis://:$(REDIS_PASSWORD)@$(REDIS_HOST):$(REDIS_PORT)/1"
- name: CELERY_RESULT_BACKEND
  value: "redis://:$(REDIS_PASSWORD)@$(REDIS_HOST):$(REDIS_PORT)/2"
{{- end -}}

{{/*
Secret-sourced env shared by api + worker. REDIS_PASSWORD is referenced first so
later $(REDIS_PASSWORD) interpolation resolves. Provider credentials are optional.
*/}}
{{- define "qufin.secretEnv" -}}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "qufin.secretName" . }}
      key: REDIS_PASSWORD
      optional: true
- name: QUFIN_FRED_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "qufin.secretName" . }}
      key: QUFIN_FRED_API_KEY
      optional: true
- name: IBM_QUANTUM_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "qufin.secretName" . }}
      key: IBM_QUANTUM_TOKEN
      optional: true
{{- end -}}
