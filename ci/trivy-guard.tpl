{{- /* trivy guard template: render one "<pkgs> <vulns>" line per Result. */ -}}
{{- /* The template context for `trivy convert --format template` IS the */ -}}
{{- /* types.Results slice, so iterate `range .`, not `range .Results`. */ -}}
{{- range . -}}
{{ len .Packages }} {{ len .Vulnerabilities }}
{{ end -}}
