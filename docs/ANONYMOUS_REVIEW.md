# Anonymous review distribution

The intended workflow is a **private GitHub source repository** plus a
read-only reviewer mirror from [Anonymous GitHub](https://anonymous.4open.science/).
The service mirrors an existing source; it is not the source repository or a
permanent DOI archive.

## Configure the mirror

1. Select the specific private BATTWISE repository with the least access needed.
2. Freeze the branch/revision supplied for review; do not rely on an evolving
   development branch to identify the reviewed version.
3. Use a neutral mirror identifier, choose the expiration policy, and include
   project-author names, affiliations, emails and identifying account names as
   anonymization terms where needed.
4. Enable images for the two synthetic workbench demonstrations under
   `docs/media/`. Verify both animations from the anonymous README after each
   update. Keep PDFs disabled and review external links for anonymity.
5. Inspect the resulting README, source tree, download, and external links from
   the anonymous view before inserting the URL into the manuscript.

Never give the mirror access to unrelated private repositories merely to share
this artifact. Authorizing new GitHub App access is an account permission step;
the repository preparation does not assume it has already been authorized.

## Preserve third-party attribution

Do not remove the EoH upstream license, copyright, or bibliographic reference.
Those are third-party provenance rather than the identity of the BATTWISE
authors. Do not include a private remote URL, account path, institution, or email
in manuscript-facing README examples.

## Publication transition

An anonymous reviewer mirror is not equivalent to a permanent public archive.
At publication, choose the public repository/release and archive the exact
relevant revision in an appropriate DOI-minting repository. Update the code
availability statement with the real version identifier, license and DOI.

## Status

The actual source URL and anonymous reviewer URL must be recorded only after
their creation and verification. Do not infer a successful mirror from a
dashboard login or a repository upload alone.
