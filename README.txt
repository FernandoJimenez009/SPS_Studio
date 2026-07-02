# Script para compilar:

python -m PyInstaller SPS_Studio.spec --clean



#Script para cambiar de version en SQL:

-- 1️⃣ Desactivar versión anterior
UPDATE dbo.AppVersionSPS
SET is_active = 0
WHERE app_name = 'SPS Studio'
  AND is_active = 1;


-- 2️⃣ Insertar nueva versión como activa
INSERT INTO dbo.AppVersionSPS (
    app_name,
    version_number,
    minimum_supported_version,
    download_url,
    is_active,
    change_log,
    message
)
VALUES (
    'SPS Studio',
    '2.19',
    '2.19',
    'https://sensata2com.sharepoint.com/sites/SMXSigSigma/Plantillas%20SPS/Forms/AllItems.aspx?viewid=65e1524e%2Dca8a%2D42fd%2Da067%2Dc88c130a8d49',
    1,
    'Add ANOVA One way and Standard Deviation Test.',
    'New version available'
);